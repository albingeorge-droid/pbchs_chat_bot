from __future__ import annotations
import re
import json
from typing import Dict, Any, List
from openai_client import GroqClient
from prompts import STANDALONE_QUESTION_PROMPT
from langsmith import traceable


PRONOUNS = {"him", "her", "them", "their", "his", "hers", "it", "its", "that", "this"}

# Domain-specific "reference" words that often refer to the last property/person
REFERENCE_TOKENS = PRONOUNS | {
    "plot",
    "property",
    "file",
    "file_no",
    "fileno",
    "owner",
    "document",
}

PERSON_PRONOUNS = {"him", "her", "them", "their", "his", "hers", "he", "she", "they"}
PROPERTY_PRONOUNS = {"it", "its", "this", "that", "these", "those"}

def _mentions_prior_context(user_query: str) -> bool:
    """
    Return True only when the user explicitly refers to earlier context.
    If False, we should NOT allow the LLM to borrow plot/road/PRA from history.
    """
    q = (user_query or "").lower()
    return bool(
        re.search(r"\b(this|it|its|these|those|above|same|previous|earlier)\b", q)
        or re.search(r"\b(same one|last one|the above|as above)\b", q)
    )



def _has_explicit_property_info(user_query: str, ner_entities: Dict[str, Any] | None) -> bool:
    ner_entities = ner_entities or {}

    # If NER already found anything explicit, do NOT inject history context
    if ner_entities.get("pra") or ner_entities.get("file_no") or ner_entities.get("file_name"):
        return True
    if ner_entities.get("plot_no") or ner_entities.get("road_no") or ner_entities.get("area"):
        return True

    q = user_query.lower()

    # PRA pattern like 28|6|Punjabi Bagh East/West
    if re.search(r"\b\d+\|\d+\|punjabi bagh (east|west)\b", q):
        return True

    # plot/road numbers present
    PLOT_ROAD_TOKEN = r"[0-9A-Za-z]+(?:[/-][0-9A-Za-z]+)*"

    if re.search(rf"\bplot\s*{PLOT_ROAD_TOKEN}\b", q) or re.search(rf"\broad\s*{PLOT_ROAD_TOKEN}\b", q):
        return True


    # file no
    if re.search(r"\bfile\s*(no|number)?\s*[:\-]?\s*\w+\b", q):
        return True

    return False

def _extract_last_property_from_history(
    history_messages: List[Dict[str, str]]
) -> Dict[str, str] | None:
    """
    Best-effort extraction of the last property mentioned in history.
    """
    if not history_messages:
        return None

    # Walk from newest → oldest
    for msg in reversed(history_messages):
        text = (msg.get("content") or "").strip()
        if not text:
            continue

        # ✅ NEW: Remove duplicated words like "plot plot number" → "plot number"
        text = re.sub(r'\b(plot|property|file|road)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

        # 1) PRA pattern
        m_pra = re.search(
            r"(\d+\|\d+\|Punjabi Bagh (?:East|West))",
            text,
            flags=re.IGNORECASE,
        )
        if m_pra:
            pra = m_pra.group(1)
            parts = pra.split("|")
            result: Dict[str, str] = {"pra": pra}
            if len(parts) == 3:
                result["plot_no"] = parts[0]
                result["road_no"] = parts[1]
                result["area"] = parts[2]
            return result
        
        PLOT_ROAD_TOKEN = r"[0-9A-Za-z]+(?:[/-][0-9A-Za-z]+)*"

        # 2) "plot X ... road Y" pattern
        m_pr = re.search(
            rf"plot\s+(?:number\s+)?({PLOT_ROAD_TOKEN}).*?road\s+(?:number\s+)?({PLOT_ROAD_TOKEN})",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if m_pr:
            plot_no, road_no = m_pr.group(1), m_pr.group(2)
            result: Dict[str, str] = {"plot_no": plot_no, "road_no": road_no}
            m_area = re.search(
                r"Punjabi Bagh\s+(East|West)",
                text,
                flags=re.IGNORECASE,
            )
            if m_area:
                result["area"] = f"Punjabi Bagh {m_area.group(1).title()}"
            return result

        # 2b) Text-based road names like "East Avenue Road", "North West Avenue Road"
        m_pr_text = re.search(
            rf"plot\s+(?:number\s+)?({PLOT_ROAD_TOKEN}).*?((?:\w+\s+)*\w+\s+[Rr]oad)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if m_pr_text:
            plot_no = m_pr_text.group(1)
            road_no = m_pr_text.group(2).strip()
            result: Dict[str, str] = {"plot_no": plot_no, "road_no": road_no}
            m_area = re.search(
                r"Punjabi Bagh\s+(East|West)",
                text,
                flags=re.IGNORECASE,
            )
            if m_area:
                result["area"] = f"Punjabi Bagh {m_area.group(1).title()}"
            return result

    return None
def _resolve_vague_property_with_history(
    user_query: str,
    history_messages: List[Dict[str, str]],
    ner_entities: Dict[str, Any] | None,
) -> str:
    """
    If the query is vague ("this plot/property") use history to fill the property.

    This implements a deterministic version of the rules in
    STANDALONE_QUESTION_PROMPT so we're not fully dependent on
    the LLM following the prompt perfectly.
    """
    user_query = re.sub(r'\b(plot|property|file|road)\s+\1\b', r'\1', user_query, flags=re.IGNORECASE)

    # If user already gave explicit identifiers, don't touch it.
    if _has_explicit_property_info(user_query, ner_entities):
        return user_query

    lower_q = user_query.lower()

    # Only try to resolve if user is clearly referring to *this* property/file.
    if not any(
        phrase in lower_q
        for phrase in (
            "this plot",
            "this property",
            "this file",
            "that plot",
            "that property",
            "that file",
        )
    ):
        return user_query

    prop = _extract_last_property_from_history(history_messages)
    if not prop:
        return user_query

    # Build a concrete identifier
    if prop.get("pra"):
        ident = f"property {prop['pra']}"
    elif prop.get("plot_no") and prop.get("road_no") and prop.get("area"):
        ident = (
            f"plot {prop['plot_no']} on road {prop['road_no']} in {prop['area']}"
        )
    elif prop.get("plot_no") and prop.get("road_no"):
        ident = f"plot {prop['plot_no']} on road {prop['road_no']}"
    else:
        return user_query

    # Replace "this plot/property/file" with the concrete identifier
    resolved = re.sub(
        r"\b(this|that)\s+(plot|property|file)\b",
        ident,
        user_query,
        flags=re.IGNORECASE,
    )
    return resolved


def _inject_focus_if_needed(
    user_query: str,
    focus_property: Dict[str, str] | None,
    focus_person: str | None,
    ner_entities: Dict[str, Any] | None = None,
) -> str:
    # If user already specified plot/road/PRA/file etc. → do not inject anything
    if _has_explicit_property_info(user_query, ner_entities):
        return user_query

    # If no focus, nothing to inject
    if not (focus_property or focus_person):
        return user_query

    tokens = set(re.sub(r"[?.!,]", " ", user_query.lower()).split())

    suffixes = []

    # Inject property ONLY if user used property pronouns and is missing explicit identifiers
    if focus_property and any(t in tokens for t in PROPERTY_PRONOUNS | {"property", "plot", "file"}):
        if focus_property.get("pra"):
            suffixes.append(f"for property PRA {focus_property['pra']}")
        elif focus_property.get("file_name"):
            suffixes.append(f"for file name {focus_property['file_name']}")

    # Inject person ONLY if user used person pronouns (him/her/them etc.)
    if focus_person and any(t in tokens for t in PERSON_PRONOUNS):
        suffixes.append(f"for person {focus_person}")

    if suffixes:
        return f"{user_query.strip()} ({', '.join(suffixes)})"
    return user_query


def _normalize_property_words_to_plot(text: str) -> str:
    """
    Replace 'property/properties' with 'plot/plots' (case-aware).
    Used to standardize wording in user-facing outputs.
    """
    if not text:
        return text

    def repl(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower == "property":
            base = "plot"
        else:  # 'properties'
            base = "plots"

        # keep capitalization style
        if word.isupper():
            return base.upper()
        if word[0].isupper():
            return base.capitalize()
        return base

    return re.sub(r"\bproperties\b|\bproperty\b", repl, text, flags=re.IGNORECASE)

@traceable(run_type="chain", name="build_standalone_question")
def build_standalone_question(
    llm: GroqClient,
    raw_query: str,
    history_messages: List[Dict[str, str]],
    ner_entities: Dict[str, Any],
    focus_property: Dict[str, str] | None,
    focus_person: str | None,
) -> Dict[str, str]:
    """
    Returns dict with keys: language, normalized_query, standalone_question
    """
    # Step 0: deterministically resolve vague "this plot/property" using history
    resolved_query = _resolve_vague_property_with_history(
        raw_query,
        history_messages,
        ner_entities,
    )

    # Step 1: optionally inject focus property/person (existing logic)
    patched_query = _inject_focus_if_needed(
        resolved_query,
        focus_property,
        focus_person,
        ner_entities,
    )

    # Only pass history to the LLM if the user explicitly refers to prior context
    use_history = _mentions_prior_context(raw_query)

    history_for_prompt = history_messages[-6:] if use_history else []
    history_json = json.dumps(history_for_prompt, ensure_ascii=False, indent=2)

    ner_json = json.dumps(ner_entities or {}, ensure_ascii=False, indent=2)

    user_prompt = STANDALONE_QUESTION_PROMPT.format(
        history_json=history_json,
        user_query=patched_query,
        ner_json=ner_json,
    )

    result = llm.generate_json(
        system_prompt=(
            "You normalize Punjabi Bagh property questions and produce standalone English versions.\n"
            "You MUST return exactly one JSON object with the keys "
            "\"language\", \"normalized_query\", and \"standalone_question\".\n"
            "If the provided user query already contains any explicit property identifiers "
            "(for example a PRA like '28|6|Punjabi Bagh East/West', plot number, road number, "
            "file number or name, or an area like 'Punjabi Bagh East/West', including inside "
            "parentheses such as '(for property PRA 30|14|Punjabi Bagh East)'), you MUST "
            "preserve those identifiers in BOTH normalized_query and standalone_question.\n"
            "You MUST NOT replace explicit identifiers with vague phrases such as "
            "'this plot', 'this property', or 'that file' in standalone_question.\n"
            "When history or NER lets you recover a specific property for a vague request, "
            "standalone_question must mention that property explicitly, not with pronouns.\n"
            "If the current user query does NOT contain explicit reference words like "
            "'this', 'it', 'above', 'same', 'previous', do NOT use chat history to infer "
            "any plot/road/PRA/file identifiers.\n"
            "Return only the JSON object, with no extra text."
        ),
        user_prompt=user_prompt,
        max_tokens=250,
        temperature=0.1,  # you can make this 0.0 if you want it even more strict
    )



    normalized = result.get("normalized_query", patched_query)

    model_standalone = result.get("standalone_question")
    if not model_standalone:
        standalone = patched_query
    else:
        standalone = model_standalone
        # If the model's standalone question still has no explicit property info
        # but our patched_query DOES, fall back to patched_query.
        if (
            not _has_explicit_property_info(standalone, ner_entities)
            and _has_explicit_property_info(patched_query, ner_entities)
        ):
            standalone = patched_query

    # 🔽 NEW: enforce 'plot/plots' wording in outputs
    normalized = _normalize_property_words_to_plot(normalized)
    standalone = _normalize_property_words_to_plot(standalone)

    # Guardrail: if the user did NOT reference prior context and did NOT provide identifiers,
    # do NOT allow the model to inject plot/road/PRA from history.
    if (not use_history) and (not _has_explicit_property_info(raw_query, ner_entities)):
        if _has_explicit_property_info(standalone, None):
            standalone = normalized


    return {
        "language": result.get("language", "english"),
        "normalized_query": normalized,
        "standalone_question": standalone,
    }