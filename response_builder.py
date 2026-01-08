from __future__ import annotations
from typing import List, Dict, Any
import json

from .openai_client import GroqClient
from .prompts import FINAL_RESPONSE_SYSTEM_PROMPT
from langsmith import traceable

HIDDEN_FIELDS = {
    "id",
    "property_id",
    "sale_deed_id",
    "buyer_id",
    "seller_id",
    "person_id",
    "file_no",
    "file_name",
    "qc_status",
    "flag",
    "status",
}

def _strip_hidden_fields(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove internal / housekeeping fields from SQL rows
    before sending them to the LLM.
    """
    cleaned: List[Dict[str, Any]] = []

    for row in rows:
        # if it's not a dict, just keep it as-is
        if not isinstance(row, dict):
            cleaned.append(row)
            continue

        cleaned.append(
            {
                k: v
                for k, v in row.items()
                # drop explicit hidden fields AND any "*_id" columns
                if k not in HIDDEN_FIELDS and not k.endswith("_id")
            }
        )

    return cleaned


@traceable(run_type="chain", name="build_final_answer")
def build_final_answer(
    llm: GroqClient,
    user_query: str,
    standalone_question: str,
    sql_query: str,
    sql_rows: List[Dict[str, Any]],
    history_messages: List[Dict[str, str]],
) -> str:
    """
    Use groq to produce a user-facing explanation of the SQL results.
    """
    safe_sql_rows = _strip_hidden_fields(sql_rows)
    rows_json = json.dumps(
    safe_sql_rows,
    ensure_ascii=False,
    indent=2,
    default=str,  # <-- this is the important bit
)
    # hist_json = json.dumps(history_messages[-4:], ensure_ascii=False, indent=2)

    user_prompt = f"""
User's original question:
{user_query}

Standalone question used for SQL:
{standalone_question}

Executed SQL query:
{sql_query}

Total rows returned: {len(sql_rows)}

Sample of result rows (JSON):
{rows_json}

Now write a concise explanation in plain English.
If there are no rows, politely say that no matching records were found and,
where possible, suggest how the user might refine the query.
"""

    return llm.generate_text(
        system_prompt=FINAL_RESPONSE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=600,
        temperature=0.2,
    )
