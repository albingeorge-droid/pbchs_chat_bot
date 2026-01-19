from __future__ import annotations
from typing import List, Dict, Any

from langsmith import traceable

from openai_client import GroqClient
from prompts import QUERY_CLASSIFICATION_SYSTEM_PROMPT
import re


@traceable(run_type="chain", name="classify_query")
def classify_property_query(
    llm: GroqClient,
    user_query: str,
    history_messages: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """
    Classify the user query as one of:
    - "property_talk"
    - "small_talk"
    - "irrelevant_question"
    """

    # 🔹 1) Simple keyword override to avoid misclassifying domain questions
    q = (user_query or "").lower()
    # 🔒 0) Hard block any data-modifying intent -> ALWAYS irrelevant_question
    mutation_pattern = r"\b(delete|update|append|remove|drop|insert|edit|change|modify)\b"
    if re.search(mutation_pattern, q):
        return {
            "label": "irrelevant_question",
            "reason": "Query attempts to modify or edit data, which is not allowed.",
        }

    property_keywords = [
        "plot",
        "plots",
        "property",
        "properties",
        "pra",
        "road",
        "file no",
        "file number",
        "file_name",
        "current owner",
        "owner",
        "owners",
        "sale deed",
        "transaction",
        "transactions",
        "society member",
        "society membership",
        "share certificate",
        "club member",
        "club membership",
        "dob",
        "date of birth",
        "birthday",
        "birthdays",
        "born",
        "email",
        "occupation",
        "work",
        "works",
        "phone",
        "phone number",
        "phone numbers",
        "mobile",
        "mobile number",
        "mobile numbers",
        "address",
        "addresses",
        "pan",
        "aadhaar",
        "pan number",
        "aadhaar number",
        "pan card",
        "aadhaar card",
        "pan card number",
        "aadhaar card number",
        "pan card number",
        "aadhaar card number",
    ]
    if any(kw in q for kw in property_keywords):
        return {
            "label": "property_talk",
            "reason": "Heuristic: query contains property-related keywords.",
        }

    # 🔹 2) Normal LLM-based classification
    history_text = ""
    if history_messages:
        # keep last few messages only, as before
        history_text = str(history_messages[-6:])

    user_prompt = f"""
You must answer with a JSON object with fields "label" and "reason".
Return only the JSON.

User message:
{user_query}

Recent conversation history (may be empty):
{history_text or "[]"}
""".strip()

    result = llm.generate_json(
        system_prompt=QUERY_CLASSIFICATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=200,
        temperature=0.0,
    )

    label = (result.get("label") or "property_talk").strip()
    reason = result.get("reason", "")
    return {"label": label, "reason": reason}
