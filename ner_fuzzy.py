from __future__ import annotations
from typing import Dict, Any
from openai_client import GroqClient
from langsmith import traceable


NER_SYSTEM_PROMPT = """
You extract entities for a property ownership database (Punjabi/Punjabhi Bagh Housing Society).

Return ONLY a JSON object with these optional keys (omit keys you cannot infer):

- "pra": string | null
- "file_name": string | null
- "file_no": string | null
- "plot_no": string | null
- "road_no": string | null
- "area": string | null                 # only: "Punjabhi Bagh East" or "Punjabhi Bagh West"
- "person": list[string] | null
- "year_from": string | null            # YYYY or DD/MM/YYYY if user specifies
- "year_to": string | null
- "intent": string | null               # one of: current_owner, ownership_history, transactions, aggregate_stats, generic_sql

Rules:
- Output MUST be valid JSON only. No markdown, no explanation.
- Do NOT hallucinate values. If unknown, use null or omit the key.
- "person": capture ANY human-name-like tokens in the user query, even if only a single
  first name is given (e.g. "neelam"). If at least one such token appears, ALWAYS include the
  "person" key with a list of strings. Only omit "person" when the query clearly
  contains no person at all.

- "area": normalize to exactly "Punjabhi Bagh East" or "Punjabhi Bagh West" when user implies east/west (e.g., "PB West", "Punjabi Bagh West", "Punjabhi Bagh West").
- Keep plot/road numbers exactly as written (may include letters or separators like "/", "-", etc.).
"""

@traceable(run_type="chain", name="extract_property_entities")
def extract_property_entities(query: str, llm: GroqClient) -> Dict[str, Any]:
    user_prompt = f"User query: {query}\n\nReturn the JSON now."
    # ner_fuzzy.py (inside extract_property_entities)
    result = llm.generate_json(NER_SYSTEM_PROMPT, user_prompt)

    result.setdefault("pra", None)
    result.setdefault("file_name", None)
    result.setdefault("file_no", None)

    # NEW fields
    result.setdefault("plot_no", None)
    result.setdefault("road_no", None)
    result.setdefault("area", None)

    # person_name is now list[string] | null
    result.setdefault("person", None)

    result.setdefault("year_from", None)
    result.setdefault("year_to", None)
    result.setdefault("intent", "generic_sql")
    return result



def fuzzy_enrich_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for fuzzy matching:
    - e.g., fetch distinct PRA/file_name/person names from Postgres
      and use rapidfuzz to snap user spelling to closest value.

    For now, just return entities unchanged.
    """
    return entities
