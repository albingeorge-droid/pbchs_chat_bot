from __future__ import annotations
from typing import List, Dict, Any
import re

from openai_client import GroqClient
from prompts import SQL_GENERATION_SYSTEM_PROMPT, TABLE_SCHEMAS

def _build_sql_prompt(
    standalone_question: str,
    ner_entities: Dict[str, Any],
    schema_docs: List[Dict[str, Any]],
    sql_example_docs: List[Dict[str, Any]],
) -> str:
    # Use ALL table schemas instead of only top-K matches from Chroma.
    all_schema_blocks = []
    for item in TABLE_SCHEMAS:
        table_name = item.get("table", "")
        desc = item.get("description", "")
        all_schema_blocks.append(f"Table {table_name}:\n{desc}")
    schema_text = "\n\n".join(all_schema_blocks)

    # ---------- Build examples block: question + SQL ----------
    examples_lines = []
    for i, ex in enumerate(sql_example_docs, start=1):
        meta = ex.get("metadata", {}) or {}
        tables = meta.get("tables", "")
        ex_question = meta.get("question") or ex.get("document", "")
        ex_sql = meta.get("sql") or ""

        if not ex_question and not ex_sql:
            continue

        lines = [f"Example {i} (tables: {tables}):"]
        if ex_question:
            lines.append(f"Question: {ex_question}")
        if ex_sql:
            lines.append("SQL:")
            lines.append(ex_sql)

        examples_lines.append("\n".join(lines))

    examples_text = "\n\n".join(examples_lines)

    ner_text = "\n".join(
        f"- {k}: {v}" for k, v in ner_entities.items() if v not in (None, "", [])
    )

    prompt = f"""
You will write a PostgreSQL SELECT query for the Property Ownership database.

User standalone question:
{standalone_question}

Extracted entities / hints:
{ner_text or '(none)'}

Relevant schema snippets:
{schema_text or '(none)'}

Relevant SQL examples:
{examples_text or '(none)'}

Return ONLY the final PostgreSQL SELECT statement, ending with a semicolon.
"""
    return prompt.strip()



def clean_sql(response_text: str) -> str:
    # Strip markdown fences
    sql = re.sub(r"```sql", "", response_text, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)
    sql = sql.strip()
    # Take only the first statement (if user returns multiple)
    if ";" in sql:
        sql = sql.split(";")[0] + ";"
    if not sql.endswith(";"):
        sql += ";"
    return sql.strip()

def generate_sql(
    llm: GroqClient,
    standalone_question: str,
    ner_entities: Dict[str, Any],
    schema_matches: List[Dict[str, Any]],
    sql_example_matches: List[Dict[str, Any]],
) -> str:
    user_prompt = _build_sql_prompt(
        standalone_question,
        ner_entities,
        schema_matches,
        sql_example_matches,
    )
    raw = llm.generate_text(
        system_prompt=SQL_GENERATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=400,
        temperature=0.0,
    )
    return clean_sql(raw)
