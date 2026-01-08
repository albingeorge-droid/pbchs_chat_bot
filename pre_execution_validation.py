from __future__ import annotations

from typing import Dict, Any, List
import re

from sqlglot import parse_one, expressions as exp
from sqlglot.errors import ParseError
from openai_client import GroqClient
from prompts import SQL_GENERATION_SYSTEM_PROMPT

# -----------------------------
# Basic config / allow-lists
# -----------------------------

DANGEROUS_KEYWORDS = {
    "DELETE",
    "UPDATE",
    "INSERT",
    "DROP",
    "TRUNCATE",
    "ALTER",
    "CREATE",
}

# Whitelist of tables we allow the LLM to touch
ALLOWED_TABLES: set[str] = {
    "properties",
    "property_addresses",
    "persons",
    "ownership_records",
    "ownership_sellers",
    "current_owners",
    "current_owner_sellers",
    "sale_deeds",
    "construction_details",
    "legal_details",
    "share_certificates",
    "club_memberships",
    "misc_documents",
}

# Per-table allowed columns, based on TABLE_SCHEMAS
ALLOWED_COLUMNS: Dict[str, set[str]] = {
    "properties": {
        "id",
        "pra",
        "file_no",
        "file_name",
        "file_link",
        "qc_status",
    },
    "property_addresses": {
        "id",
        "property_id",
        "plot_no",
        "road_no",
        "street_name",
        "initial_plot_size",
        "source_page",
        "flag",
    },
    "persons": {
        "id",
        "pra",
        "name",
        "dob",
        "family_members",
        "address",
        "phone_number",
        "email",
        "pan",
        "aadhaar",
        "img_link",
        "occupation",
        "source_page",
        "person_source",
        "flag",
    },
    "ownership_records": {
        "id",
        "property_id",
        "buyer_id",
        "sale_deed_id",
        "transfer_type",
        "buyer_portion",
        "total_stamp_duty_paid",
        "notes",
        "source_page",
        "flag",
    },
    "ownership_sellers": {
        "ownership_id",
        "person_id",
    },
    "current_owners": {
        "id",
        "property_id",
        "buyer_id",
        "buyer_portion",
        "source_page",
        "flag",
    },
    "current_owner_sellers": {
        "current_owner_id",
        "person_id",
    },
    "sale_deeds": {
        "id",
        "person_id",
        "property_id",
        "sale_deed_no",
        "book_no",
        "page_no",
        "signing_date",
        "registry_status",
        "owners_portion_sold",
        "total_property_portion_sold",
        "source_page",
        "pdf_link",
        "flag",
    },
    "construction_details": {
        "id",
        "property_id",
        "coverage_built_up_area",
        "circle_rate_colony",
        "land_price_per_sqm",
        "construction_price_per_sqm",
        "total_covered_area",
        "source_page",
        "pdf_link",
        "flag",
    },
    "legal_details": {
        "id",
        "property_id",
        "registrar_office",
        "court_cases",
        "source_page",
        "pdf_link",
        "flag",
    },
    "share_certificates": {
        "id",
        "certificate_number",
        "property_id",
        "member_id",
        "date_of_transfer",
        "date_of_ending",
        "notes",
        "source_page",
        "pdf_link",
        "flag",
    },
    "club_memberships": {
        "id",
        "member_id",
        "property_id",
        "allocation_date",
        "membership_end_date",
        "membership_number",
        "source_page",
        "pdf_link",
        "flag",
    },
    "misc_documents": {
        "id",
        "property_id",
        "pra",
    },
}




class SQLValidationError(Exception):
    """Raised when a generated SQL query is unsafe or invalid."""
    pass


# -----------------------------
# Low-level validation helpers
# -----------------------------


def _clean_sql_one_statement(sql: str) -> str:
    """
    Strip fences & enforce *single* statement.

    This is deliberately strict: multiple semicolons are rejected.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```sql", "", sql, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    # No empty queries
    if not cleaned:
        raise SQLValidationError("Empty SQL query from LLM")

    # 4) Stronger clean_sql: block multiple statements
    if cleaned.count(";") > 1:
        raise SQLValidationError(
            "LLM produced multiple SQL statements; blocking."
        )

    # Ensure we end in ';' (only one at this point)
    if not cleaned.endswith(";"):
        cleaned += ";"

    return cleaned.strip()


def _cheap_keyword_guard(sql: str) -> None:
    upper = sql.upper()
    if not upper.lstrip().startswith("SELECT"):
        raise SQLValidationError("Only SELECT queries are allowed.")

    for kw in DANGEROUS_KEYWORDS:
        if kw in upper:
            raise SQLValidationError(f"Disallowed keyword '{kw}' found in SQL.")



def _enforce_limit(sql: str, default_limit: int = 100) -> str:
    """
    Ensure every SELECT has a LIMIT, *except* when it's an aggregate query
    (COUNT, MAX, MIN, AVG, SUM, etc.).

    Rules:
    - If query already has LIMIT -> keep it (just normalize & re-add ';').
    - If query has aggregates -> do NOT add a LIMIT.
    - Otherwise -> add 'LIMIT <default_limit>' at the end.
    """
    raw = sql.strip()

    # Strip one trailing semicolon before parsing, if present
    if raw.endswith(";"):
        raw = raw[:-1].strip()

    # If there's already a LIMIT anywhere, just normalize & re-semicolon
    if re.search(r"\bLIMIT\b", raw, flags=re.IGNORECASE):
        ast = parse_one(raw, read="postgres")
        normalized = ast.sql(dialect="postgres").strip()
        return normalized + ";"

    # Parse for aggregate detection
    try:
        ast = parse_one(raw, read="postgres")
    except ParseError as e:
        raise SQLValidationError(f"SQL parse error while enforcing LIMIT: {e}") from e

    # Only support SELECT-like top-level queries
    if not isinstance(ast, (exp.Select, exp.Subquery, exp.With)):
        raise SQLValidationError("Top-level query must be a SELECT.")

    # Find the main SELECT node
    if isinstance(ast, exp.With):
        select = ast.this          # query after WITH
    elif isinstance(ast, exp.Subquery):
        select = ast.this
    else:
        select = ast

    if not isinstance(select, exp.Select):
        raise SQLValidationError("Top-level query must be a SELECT.")

    # Detect if the query uses aggregate functions
    has_aggregate = any(
        isinstance(node, exp.AggFunc) for node in select.find_all(exp.AggFunc)
    )

    normalized = ast.sql(dialect="postgres").strip()

    # Aggregate queries: don't add LIMIT
    if has_aggregate:
        return normalized + ";"

    # Non-aggregate & no LIMIT: append ' LIMIT 100;'
    return f"{normalized} LIMIT {default_limit};"




def _table_alias_name(t: exp.Table) -> str | None:
    """
    Extract alias name from a sqlglot Table node.
    Works for: FROM ownership_records AS T1, JOIN ownership_sellers T5, etc.
    """
    alias = t.args.get("alias")
    if not alias:
        return None

    # sqlglot usually stores alias as exp.TableAlias(this=exp.Identifier(...))
    alias_ident = getattr(alias, "this", None)
    if alias_ident is not None and getattr(alias_ident, "name", None):
        return alias_ident.name

    # fallback: try common attribute names
    if getattr(alias, "name", None):
        return alias.name

    return None

def _collect_select_aliases(ast: exp.Expression) -> set[str]:
    """
    Collect aliases defined in SELECT lists, e.g.:

        SELECT sd.signing_date->>0 AS ownership_date, ...

    so that we can allow ORDER BY ownership_date, etc.
    """
    aliases: set[str] = set()

    # Look for expression aliases (not table aliases)
    for alias_node in ast.find_all(exp.Alias):
        alias_expr = alias_node.args.get("alias")
        if not alias_expr:
            continue

        # Similar to _table_alias_name: alias_expr.should be an Identifier
        alias_ident = getattr(alias_expr, "this", None)
        alias_name = getattr(alias_ident, "name", None)
        if alias_name:
            aliases.add(alias_name)

    return aliases

def _collect_subquery_aliases(ast: exp.Expression) -> set[str]:
    """
    Collect aliases for derived tables / subqueries, e.g.:

        JOIN (
          SELECT ...
        ) AS latest ON ...

    so we can allow columns like latest.property_id.
    """
    aliases: set[str] = set()

    for sub in ast.find_all(exp.Subquery):
        alias = sub.args.get("alias")
        if not alias:
            continue

        alias_ident = getattr(alias, "this", None)
        alias_name = getattr(alias_ident, "name", None)
        if alias_name:
            aliases.add(alias_name)

    return aliases


def _guard_tables_and_columns(sql: str) -> None:
    """
    Whitelist tables & columns using sqlglot AST.
    Handles table aliases (T1/T2/...) correctly and FAILS CLOSED.
    """
    ast = parse_one(sql, read="postgres")

    # ---- Build alias -> real_table map ----
    alias_map: Dict[str, str] = {}
    for t in ast.find_all(exp.Table):
        real = t.name
        alias = _table_alias_name(t)

        # map real->real always
        alias_map[real] = real
        # map alias->real if alias exists
        if alias:
            alias_map[alias] = real

    # ---- Collect SELECT expression aliases (e.g. ownership_date) ----
    select_aliases = _collect_select_aliases(ast)

    # ---- ✅ NEW: collect subquery / derived-table aliases (e.g. latest) ----
    subquery_aliases = _collect_subquery_aliases(ast)

    # ---- Check tables (real names only) ----
    tables = {t.name for t in ast.find_all(exp.Table)}
    for t in tables:
        if t not in ALLOWED_TABLES:
            raise SQLValidationError(f"Table '{t}' is not in the allowed whitelist.")

    # ---- Check columns ----
    for col in ast.find_all(exp.Column):
        qualifier = col.table or None   # this might be alias like "orc" or "latest"
        name = col.name

        if qualifier is None:
            # ✅ Allow bare usage of SELECT aliases like "ownership_date"
            if name in select_aliases:
                continue

            # bare column name – allow if it exists in ANY table whitelist
            if not any(name in cols for cols in ALLOWED_COLUMNS.values()):
                raise SQLValidationError(f"Bare column '{name}' is not allowed.")
            continue

        # resolve alias -> real table (or derived-table alias)
        real_table = alias_map.get(qualifier)
        if not real_table:
            # ✅ NEW: allow columns from derived-table / subquery aliases, e.g. latest.property_id
            if qualifier in subquery_aliases:
                # The subquery body itself is checked against the same table/column whitelist,
                # so we don't need to resolve its "table" here.
                continue

            # Anything else is unknown / unsafe
            raise SQLValidationError(
                f"Unknown table/alias '{qualifier}' used in column '{qualifier}.{name}'."
            )

        allowed = ALLOWED_COLUMNS.get(real_table)
        if allowed is None:
            raise SQLValidationError(f"Table '{real_table}' has no allowed-column config.")

        if name not in allowed:
            raise SQLValidationError(f"Column '{real_table}.{name}' is not allowed.")


def clean_and_validate_sql(sql: str) -> tuple[str, Dict[str, Any]]:
    """
    Full validation pipeline.

    Steps:
    - Strip markdown fences + enforce single statement
    - Cheap keyword guard (reject non-SELECT / dangerous verbs)
    - AST-based table & column whitelist
    - Enforce LIMIT 100 if missing / normalize LIMIT

    Returns:
    - final safe SQL string
    - debug dict describing what was run
    """
    debug: Dict[str, Any] = {
        "original_sql": sql,
        "checks_applied": [],
    }

    # 1) Single-statement + fence cleaning
    debug["checks_applied"].append("clean_single_statement")
    cleaned = _clean_sql_one_statement(sql)
    debug["cleaned_sql"] = cleaned

    # 2) Keyword guard
    debug["checks_applied"].append("keyword_guard")
    _cheap_keyword_guard(cleaned)

    # 3) Table & column whitelist
    debug["checks_applied"].append("table_column_whitelist")
    _guard_tables_and_columns(cleaned)

    # 4) LIMIT enforcement
    debug["checks_applied"].append("limit_enforcer")
    limited = _enforce_limit(cleaned)
    debug["final_sql"] = limited

    return limited, debug



# -----------------------------
# LLM-based repair loop
# -----------------------------



def validate_and_maybe_regenerate_sql(
    llm: GroqClient,
    sql_query: str,
    question: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Validate LLM SQL; if unsafe, ask the LLM again to fix it.

    Returns a dict (nice for LangSmith):

    {
      "sql": "<final_safe_sql>",
      "attempts": [
        {
          "attempt": 1,
          "ok": true/false,
          "error": "...",              # only if ok == False
          "original_sql": "...",
          "cleaned_sql": "...",
          "final_sql": "...",
          "checks_applied": [...]
        },
        ...
      ]
    }
    """
    last_error: str | None = None
    attempts_info: List[Dict[str, Any]] = []

    # Build a schema string we can show to the LLM so it stops inventing columns
    schema_lines: List[str] = []
    for table in sorted(ALLOWED_TABLES):
        cols = sorted(ALLOWED_COLUMNS.get(table, []))
        schema_lines.append(f"- {table}: {', '.join(cols)}")
    schema_text = "\n".join(schema_lines)

    for attempt in range(max_retries + 1):
        try:
            final_sql, debug = clean_and_validate_sql(sql_query)

            # record successful attempt
            attempts_info.append(
                {
                    "attempt": attempt + 1,
                    "ok": True,
                    **debug,
                }
            )

            return {
                "sql": final_sql,
                "attempts": attempts_info,
            }

        except SQLValidationError as e:
            last_error = str(e)

            # record failed attempt
            attempts_info.append(
                {
                    "attempt": attempt + 1,
                    "ok": False,
                    "error": last_error,
                    "original_sql": sql_query,
                }
            )

            if attempt >= max_retries:
                # no more retries -> bubble up
                raise

            # Ask the LLM to repair the query and loop again
            repair_prompt = f"""
You previously wrote this PostgreSQL SELECT query for the question:

{question}

SQL:
```sql
{sql_query}
```
It was rejected by the SQL safety validator with this error:
{last_error}

You MUST fix the query using ONLY the tables and columns in the schema below.

ALLOWED SCHEMA:
{schema_text}

Requirements:

Single SELECT statement only (no CTEs that modify data, no DDL/DML)

Use only the tables and columns listed above

Never invent new columns (e.g. do NOT use columns that are not in the lists)

Do not use INSERT/UPDATE/DELETE/CREATE/DROP/ALTER/TRUNCATE

Include a LIMIT clause (the validator may normalize it)

Return ONLY the final PostgreSQL SELECT statement, ending with a semicolon.
""".strip()

            # 🔁 Regenerate SQL using the repair prompt
            sql_query = llm.generate_text(
                system_prompt=SQL_GENERATION_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                max_tokens=400,
                temperature=0.0,
            )

    # Should be unreachable if max_retries >= 0
    raise SQLValidationError(f"Failed to produce safe SQL: {last_error or 'unknown error'}")
