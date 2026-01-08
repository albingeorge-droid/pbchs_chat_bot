from __future__ import annotations
from typing import List, Dict, Any
import json

from openai_client import GroqClient
from prompts import FINAL_RESPONSE_SYSTEM_PROMPT


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
    rows_json = json.dumps(
    sql_rows,
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
