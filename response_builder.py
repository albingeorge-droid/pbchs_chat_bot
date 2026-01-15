from __future__ import annotations
from typing import List, Dict, Any, Callable
import json

from openai_client import GroqClient
from prompts import FINAL_RESPONSE_SYSTEM_PROMPT
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

@traceable(run_type="llm", name="final_answer_llm")
def _call_final_answer_llm(
    llm: GroqClient,
    system_prompt: str,
    user_prompt: str,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """
    Thin wrapper around OPenai Client that LangSmith can see as an LLM run.
    """
    # Non-streaming path
    if on_token is None:
        return llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=600,
            temperature=0.2,
        )

    # Streaming path – still traced, but we manually call the callback
    chunks: list[str] = []
    for piece in llm.stream_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=600,
        temperature=0.2,
    ):
        chunks.append(piece)
        on_token(piece)

    return "".join(chunks)


@traceable(run_type="chain", name="build_final_answer")
def build_final_answer(
    llm: GroqClient,
    user_query: str,
    standalone_question: str,
    sql_query: str,
    sql_rows: List[Dict[str, Any]],
    history_messages: List[Dict[str, str]],
    on_token: Callable[[str], None] | None = None,
) -> str:
    """
    Use groq to produce a user-facing explanation of the SQL results.
    """
    # NEW: Only take first 15 rows for LLM processing
    limited_rows = sql_rows[:15]

    safe_sql_rows = _strip_hidden_fields(limited_rows)
    rows_json = json.dumps(
        safe_sql_rows,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    user_prompt = f"""

Standalone question used for SQL:
{standalone_question}


Total rows returned: {len(sql_rows)}

Sample of result rows (JSON):
{rows_json}

Now write a concise explanation in plain English.
If there are no rows, politely say that no matching records were found and,
where possible, suggest how the user might refine the query.
"""


    # Call the LLM (streaming or non-streaming)
    answer = _call_final_answer_llm(
        llm=llm,
        system_prompt=FINAL_RESPONSE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        on_token=on_token,
    )

    # If there are many rows, append the notice
    if len(sql_rows) > 15:
        extra_sentence = (
            "\n\nThere are more than 15 records, so not all the records "
            "might be displayed."
        )
        # If we’re streaming, also stream this extra bit
        if on_token is not None:
            on_token(extra_sentence)
        answer += extra_sentence

    return answer
