from __future__ import annotations
import re
from typing import TypedDict, Annotated, Sequence, Literal, Callable, Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langsmith import traceable

from openai_client import GroqClient
from memory import HistoryManager, ConversationMemory
from vector_store import PropertyVectorStore
from embedding_client import SentenceEmbeddingClient
from query_classifier import classify_property_query
from ner_fuzzy import extract_property_entities, fuzzy_enrich_entities
from standalone import build_standalone_question
from sql_generation import generate_sql
from db import run_select
from response_builder import build_final_answer
from pre_execution_validation import validate_and_maybe_regenerate_sql
from prompts import (
    SMALL_TALK_SYSTEM_PROMPT,
    OUT_OF_SCOPE_SYSTEM_PROMPT,
    SQL_GENERATION_SYSTEM_PROMPT,
    NOTE_SUMMARY_SYSTEM_PROMPT,   # NEW
)
from map import (
    parse_plot_road_from_text,
    lookup_pra_for_plot_road,
    fetch_map_for_pra,
)

from note_summary import generate_property_note_pdf  # NEW
from sqlalchemy.exc import ProgrammingError
from rapidfuzz import fuzz, process



SQL_SIMILARITY_THRESHOLD = 0.3


# Define the state that flows through the graph
class ChatbotState(TypedDict):
    # Input
    user_query: str
    
    # Intermediate results
    history_messages: list[dict[str, str]]
    classification: dict[str, str]
    ner_entities: dict
    standalone_info: dict[str, str]
    sql_matches: list[dict]
    schema_matches: list[dict]
    sql_query: str
    sql_rows: list[dict]
    
    # Output
    final_answer: str
    error: str | None

    # Optional note-summary extras
    note_pra: str | None
    note_pdf_path: str | None

    #geometry for map responses (GeoJSON features)
    geometry: list[dict] | None


class PropertyChatbotGraph:
    def __init__(self, user_id: str | None = None, thread_id: str | None = None):
        # one graph instance = one user/thread
        self.user_id = user_id or "default_user"
        self.thread_id = thread_id  # can be None; HistoryManager will fall back to user_id

        self.llm = GroqClient()
        self.history = HistoryManager()
        self.memory = ConversationMemory()
        self.embedder = SentenceEmbeddingClient()
        self.vstore = PropertyVectorStore(self.embedder)
        # Ephemeral state for the note-summary wizard (not persisted in HistoryManager)
        self.note_flow = {
            "active": False,        # are we in the middle of a note flow?
            "step": None,           # "plot", "road"
            "plot": None,
            "road": None,
        }
        
        # Optional streaming callback for final answer tokens
        self.on_token: Callable[[str], None] | None = None

        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        workflow = StateGraph(ChatbotState)
        
        # Add nodes
        workflow.add_node("load_history", self.load_history)
        workflow.add_node("classify", self.classify_query)
        workflow.add_node("handle_small_talk", self.handle_small_talk)
        workflow.add_node("handle_irrelevant", self.handle_irrelevant)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("build_standalone", self.build_standalone)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("build_answer", self.build_answer)
        workflow.add_node("save_history", self.save_history)

        # NOTE SUMMARY nodes (multi-step)
        workflow.add_node("start_note_summary", self.start_note_summary)
        workflow.add_node("collect_plot", self.collect_plot)
        workflow.add_node("collect_road", self.collect_road)

        # ✅ single-shot note summary (plot+road in one message)
        workflow.add_node("note_direct", self.note_summary_direct)

        # ✅ MAP node (single-turn)
        workflow.add_node("map_lookup", self.map_lookup)
        
        # Set entry point
        workflow.set_entry_point("load_history")
        
        # Define edges
        workflow.add_edge("load_history", "classify")
        
        # Conditional routing after classification
        workflow.add_conditional_edges(
            "classify",
            self.route_by_classification,
            {
                "small_talk": "handle_small_talk",
                "irrelevant": "handle_irrelevant",
                "property_talk": "extract_entities",
                "note_start": "start_note_summary",
                "note_plot": "collect_plot",
                "note_road": "collect_road",
                "note_direct": "note_direct",
                "map": "map_lookup",
            },
        )
        
        # Small talk and irrelevant end immediately
        workflow.add_edge("handle_small_talk", END)
        workflow.add_edge("handle_irrelevant", END)
        
        # Property talk flow
        workflow.add_edge("extract_entities", "build_standalone")
        workflow.add_edge("build_standalone", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_sql")
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "build_answer")
        workflow.add_edge("build_answer", "save_history")
        workflow.add_edge("save_history", END)

        # NOTE SUMMARY flow
        workflow.add_edge("start_note_summary", END)
        workflow.add_edge("collect_plot", END)
        workflow.add_edge("collect_road", END)
        workflow.add_edge("note_direct", END) 
        
        workflow.add_edge("map_lookup", "save_history")

        
        return workflow.compile()

    
    # Node functions
    def load_history(self, state: ChatbotState) -> ChatbotState:
        """Load conversation history for this user/thread."""
        state["history_messages"] = self.history.last_messages(
            user_id=self.user_id,
            thread_id=self.thread_id,
            k=6,
        )
        return state

    
    def classify_query(self, state: ChatbotState) -> ChatbotState:
        """Classify the user query"""
        cls = classify_property_query(
            llm=self.llm,
            user_query=state["user_query"],
            history_messages=state["history_messages"]
        )
        state["classification"] = cls
        return state

    def _is_map_trigger(self, text: str) -> bool:
        """
        Detect 'show the map...' style questions.

        Very simple heuristic: must contain 'map' and either 'plot'/'road'
        or look like 'map 30/14'.
        """
        if not text:
            return False

        base = text.lower()

        if "map" not in base:
            return False

        # strong hints
        if "plot" in base or "road" in base:
            return True

        # fallback: "map 30/14" etc
        if re.search(r"\bmap\s+(\d{1,4})\s*[/ ]\s*(\d{1,4})\b", base):
            return True

        return False




    def _is_note_summary_trigger(self, text: str, threshold: int = 80) -> bool:
        """
        Use fuzzy matching to detect 'note summary' like phrases.
        Examples it should catch:
        - 'note summary'
        - 'generate note summary'
        - 'generate not summary'
        - 'note summar'
        - 'note summery'
        """
        if not text:
            return False

        base = text.lower().strip()

        # Quick cheap checks first
        simple_triggers = [
            "note summary",
            "note summar",
            "note summery",
        ]
        for s in simple_triggers:
            if s in base:
                return True

        # Fuzzy match against a small set of variants
        candidates = [
            "note summary",
            "generate note summary",
            "generate property note summary",
            "create note summary",
            "property note summary",
            "note summar",
            "note summery",
            "generate not summary",
        ]

        for phrase in candidates:
            score = fuzz.partial_ratio(base, phrase)
            if score >= threshold:
                return True

        return False
   
    def route_by_classification(self, state: ChatbotState) -> str:
        """
        Decide where to go after classification.

        Priority:
        1) If we are mid note-summary flow, route by self.note_flow.step
        2) If this message starts a note-summary flow:
           - if it already has plot+road → note_direct
           - else → start_note_summary
        3) Otherwise use normal classifier label
        """
        user_q_raw = state["user_query"]

        # 1) Ongoing note-summary wizard: route by step and SKIP classifier
        if self.note_flow.get("active"):
            step = self.note_flow.get("step")
            if step == "plot":
                return "note_plot"
            if step == "road":
                return "note_road"

        # 2) ✅ Single-turn map query
        if self._is_map_trigger(user_q_raw):
            return "map"

        # 3) Note-summary request (fuzzy)
        if self._is_note_summary_trigger(user_q_raw):
            # Try to see if user already gave plot+road in same sentence
            plot_raw, road_raw = parse_plot_road_from_text(user_q_raw)

            if plot_raw and road_raw:
                # Single-shot: don't run the interactive wizard
                self.note_flow = {
                    "active": False,
                    "step": None,
                    "plot": None,
                    "road": None,
                }
                return "note_direct"

            # Old behaviour: start 2-step wizard
            self.note_flow = {
                "active": True,
                "step": "plot",
                "plot": None,
                "road": None,
            }
            return "note_start"

        # 4) Normal routing based on classifier label
        label = state["classification"].get("label", "property_talk").strip()
        if label == "small_talk":
            return "small_talk"
        elif label == "irrelevant_question":
            return "irrelevant"
        else:
            return "property_talk"

    def _fuzzy_match_column(
        self,
        column: str,
        raw_value: str,
        threshold: int = 65,
    ) -> str:
        """
        Fuzzy match a user-entered value against distinct values
        in property_addresses.<column> using RapidFuzz.

        column should be 'plot_no' or 'road_no'.
        """
        value = (raw_value or "").strip()
        if not value:
            return raw_value

        # Safety: only allow specific columns
        if column not in ("plot_no", "road_no"):
            return raw_value

        # Escape single quotes
        safe_column = column

        sql_distinct = f"""
SELECT DISTINCT TRIM({safe_column}) AS val
FROM property_addresses
WHERE {safe_column} IS NOT NULL
  AND TRIM({safe_column}) <> ''
LIMIT 5000;
""".strip()

        rows = run_select(sql_distinct) or []
        choices = [str(r.get("val")).strip() for r in rows if r.get("val")]

        if not choices:
            return raw_value

        # RapidFuzz best match
        result = process.extractOne(value, choices, scorer=fuzz.WRatio)
        if not result:
            return raw_value

        best_match, score, _ = result
        if score >= threshold:
            return best_match

        return raw_value

    def _fuzzy_match_person_name(
        self,
        raw_name: str,
        threshold: int = 85,
    ) -> str:
        """
        Fuzzy match a person name against persons.name using RapidFuzz.
        """
        name = (raw_name or "").strip()
        if not name:
            return raw_name

        sql_distinct = """
SELECT DISTINCT TRIM(name) AS val
FROM persons
WHERE name IS NOT NULL
  AND TRIM(name) <> ''
LIMIT 45000;
""".strip()

        rows = run_select(sql_distinct) or []
        choices = [str(r.get("val")).strip() for r in rows if r.get("val")]

        if not choices:
            return raw_name

        result = process.extractOne(name, choices, scorer=fuzz.WRatio)
        if not result:
            return raw_name

        best_match, score, _ = result
        if score >= threshold:
            return best_match

        return raw_name

    def _normalize_punjabi_bagh(self, question: str) -> str:
        """
        Normalize Punjabi Bagh East/West in the question text.

        We match only the '(Punjabi) Bagh East/West' phrase itself, so we don't
        eat words like 'properties in' or the space before the number.
        """
        if not question:
            return question

        text = question

        # Match:
        #   - 'Bagh East/West'
        #   - 'Punjabi Bagh East/West'
        #   - 'Punjabhi Bagh East/West', etc. (one word containing 'punjab')
        match = re.search(
            r"((?:[A-Za-z]*?punjab[A-Za-z]*\s+)?bagh\s+(east|west))",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return text

        span_text = match.group(1)   # e.g. "Punjabhi Bagh East"
        span_lower = span_text.lower()

        # Decide canonical form
        if "east" in span_lower:
            canonical = "Punjabi Bagh East"
        elif "west" in span_lower:
            canonical = "Punjabi Bagh West"
        else:
            return text  # safety

        # Make sure it's actually close to the canonical phrase
        score = fuzz.WRatio(span_lower, canonical.lower())
        if score < 70:
            return text

        start, end = match.span(1)
        normalized = text[:start] + canonical + text[end:]
        return normalized

    def _normalize_plot_road_patterns(
        self,
        question: str,
        ner: dict,
    ) -> tuple[str, dict]:
        """
        Convert '30/14' or '30 14' patterns into
        'plot number 30 road 14' and update ner['plot_no'] / ner['road_no'].
        """
        if not question:
            return question, ner

        text = question
        ner = ner or {}

        # 1) pattern with slash: 30/14
        match = re.search(r"\b(\d{1,4})\s*/\s*(\d{1,4})\b", text)
        if match:
            plot, road = match.group(1), match.group(2)
            replacement = f"plot number {plot} road {road}"
            text = text[: match.start()] + replacement + text[match.end() :]

            ner["plot_no"] = plot
            ner["road_no"] = road
            return text, ner

        # 2) pattern with space: 30 14
        match = re.search(r"\b(\d{1,4})\s+(\d{1,4})\b", text)
        if match:
            plot, road = match.group(1), match.group(2)
            replacement = f"plot number {plot} road {road}"
            text = text[: match.start()] + replacement + text[match.end() :]

            ner["plot_no"] = plot
            ner["road_no"] = road

        return text, ner

    def _replace_name_in_question(
        self,
        text: str,
        raw_name: str | None,
        canonical_name: str,
    ) -> str:
        """
        Ensure the question text uses the canonical person name.

        Strategy:
        1) If canonical already in text -> do nothing.
        2) Else if raw_name appears in text (any case) -> replace that substring.
        3) Else replace the trailing capitalized name-like phrase with canonical.
        """
        if not text or not canonical_name:
            return text

        # 1) Already present
        if canonical_name in text:
            return text

        # 2) Replace raw_name if present
        if raw_name:
            if raw_name in text:
                return text.replace(raw_name, canonical_name)

            pattern_ci = re.compile(re.escape(raw_name), flags=re.IGNORECASE)
            if pattern_ci.search(text):
                return pattern_ci.sub(canonical_name, text)

        # 3) Replace trailing capitalized phrase (typical for names)
        #    e.g. "What is the date of birth of Chitranjn?"
        m = re.search(r"([A-Z][a-zA-Z']*(?:\s+[A-Z][a-zA-Z']*)*)[^\w]*?$", text)
        if m:
            start, end = m.span(1)
            return text[:start] + canonical_name + text[end:]

        # Fallback: append canonical name in parentheses
        return f"{text} ({canonical_name})"


    def _apply_fuzzy_plot_road(
        self,
        question: str,
        ner: dict,
    ) -> tuple[str, dict]:
        """
        Use DB + RapidFuzz to snap plot_no and road_no to canonical values
        and reflect them back into the question text.
        """
        if not question:
            return question, ner

        text = question
        ner = ner or {}

        plot_val = ner.get("plot_no")
        if isinstance(plot_val, list):
            plot_val = plot_val[0] if plot_val else None

        road_val = ner.get("road_no")
        if isinstance(road_val, list):
            road_val = road_val[0] if road_val else None

        # Fuzzy plot
        if plot_val:
            matched_plot = self._fuzzy_match_column("plot_no", str(plot_val))
            if matched_plot and matched_plot != plot_val:
                text = re.sub(
                    rf"\b{re.escape(str(plot_val))}\b",
                    matched_plot,
                    text,
                )
                ner["plot_no"] = matched_plot

        # Fuzzy road
        if road_val:
            matched_road = self._fuzzy_match_column("road_no", str(road_val))
            if matched_road and matched_road != road_val:
                text = re.sub(
                    rf"\b{re.escape(str(road_val))}\b",
                    matched_road,
                    text,
                )
                ner["road_no"] = matched_road

        return text, ner

    def _apply_fuzzy_person_names(
        self,
        question: str,
        ner: dict,
    ) -> tuple[str, dict]:
        """
        Use DB + RapidFuzz to normalize person_name values and
        ensure the canonical name appears in the question text.
        """
        if not question:
            return question, ner

        text = question
        ner = ner or {}
        
        # ✅ NEW: skip fuzzy person-name normalization for surname queries
        lower_q = text.lower()
        if "last name" in lower_q or "surname" in lower_q:
            # Keep whatever NER extracted (e.g. "Kohli") and don't touch the text.
            return text, ner

        persons = ner.get("person")
        if not persons:
            return text, ner

        # Ensure list
        if isinstance(persons, str):
            person_list = [persons]
            single = True
        else:
            try:
                person_list = list(persons)
                single = False
            except TypeError:
                person_list = []
                single = False

        updated_persons: list[str] = []

        for raw_name in person_list:
            if not raw_name:
                updated_persons.append(raw_name)
                continue

            canonical = self._fuzzy_match_person_name(str(raw_name))
            if not canonical:
                canonical = str(raw_name)

            # Make sure the question text uses the canonical name
            text = self._replace_name_in_question(text, str(raw_name), canonical)

            updated_persons.append(canonical)

        # Write back to ner in same shape it came in
        if single:
            ner["person"] = updated_persons[0] if updated_persons else persons
        else:
            ner["person"] = updated_persons

        return text, ner

    def _postprocess_standalone_question(
        self,
        question: str,
        ner: dict,
    ) -> tuple[str, dict]:
        """
        Apply all post-processing steps on the standalone question:
        - normalize 30/14 or 30 14 → 'plot number 30 road 14'
        - normalize Punjabi Bagh East/West casing
        - fuzzy match plot/road numbers against DB
        - fuzzy match person names against DB
        """
        if not question:
            return question, ner

        ner = ner or {}

        # 1) 30/14 or 30 14 → plot number 30 road 14
        text, ner = self._normalize_plot_road_patterns(question, ner)

        # 2) Punjabi Bagh East / West
        text = self._normalize_punjabi_bagh(text)

        # 3) Fuzzy plot / road using DB
        text, ner = self._apply_fuzzy_plot_road(text, ner)

        # 4) Fuzzy person names using DB
        text, ner = self._apply_fuzzy_person_names(text, ner)

        return text, ner

    
    def handle_small_talk(self, state: ChatbotState) -> ChatbotState:
        """Handle small talk without SQL"""
        answer = self.llm.generate_text(
            system_prompt=SMALL_TALK_SYSTEM_PROMPT,
            user_prompt=state["user_query"],
            max_tokens=300,
            temperature=0.7
        )
        state["final_answer"] = answer
        state["sql_query"] = "-- NO SQL (small_talk)"
        state["sql_rows"] = []
        state["geometry"] = None
        return state
    
    def handle_irrelevant(self, state: ChatbotState) -> ChatbotState:
        """Handle irrelevant questions"""
        answer = self.llm.generate_text(
            system_prompt=OUT_OF_SCOPE_SYSTEM_PROMPT,
            user_prompt=state["user_query"],
            max_tokens=200,
            temperature=0.2
        )
        state["final_answer"] = answer
        state["sql_query"] = "-- NO SQL (irrelevant_question)"
        state["sql_rows"] = []
        state["geometry"] = None
        return state
    
    def start_note_summary(self, state: ChatbotState) -> ChatbotState:
        """
        First step when user says 'note summary' / 'generate note summary'.
        Ask for plot number.
        """
        # Ensure note flow is initialized
        self.note_flow["active"] = True
        self.note_flow["step"] = "plot"
        self.note_flow["plot"] = None
        self.note_flow["road"] = None

        answer = (
            "To generate a property note summary I just need two details:\n"
            "Step 1: Please tell me the plot number."
        )
        state["final_answer"] = answer
        state["sql_query"] = "-- NOTE_SUMMARY_FLOW"
        state["sql_rows"] = []
        # ⚠️ Do NOT save history here
        return state

    
    def collect_plot(self, state: ChatbotState) -> ChatbotState:
        """
        Second turn: user gives plot number.
        Apply fuzzy matching against property_addresses.plot_no.
        """
        raw_plot = state["user_query"].strip()
        matched_plot = self._fuzzy_match_column("plot_no", raw_plot, threshold=80)

        self.note_flow["plot"] = matched_plot
        self.note_flow["step"] = "road"

        if matched_plot != raw_plot:
            line = f"Got it, I interpreted plot '{raw_plot}' as '{matched_plot}' based on existing records."
        else:
            line = f"Got it, plot number is {matched_plot}."

        answer = (
            f"{line}\n"
            "Step 2: Please tell me the road number."
        )
        state["final_answer"] = answer
        state["sql_query"] = "-- NOTE_SUMMARY_FLOW"
        state["sql_rows"] = []
        # ⚠️ Do NOT save history here
        return state


    def collect_road(self, state: ChatbotState) -> ChatbotState:
        """
        Final step: user gives road number.
        - Fuzzy match road_no against DB
        - Use (plot, road) to look up PRA
        - Generate the note PDF
        """
        raw_road = state["user_query"].strip()
        matched_road = self._fuzzy_match_column("road_no", raw_road, threshold=65)
        self.note_flow["road"] = matched_road

        plot = (self.note_flow.get("plot") or "").strip()
        road = matched_road.strip()

        # Basic safety for SQL literal
        safe_plot = plot.replace("'", "''")
        safe_road = road.replace("'", "''")

        # 1) Look up PRA using plot + road
        sql_pra_lookup = f"""
SELECT p.pra
FROM properties p
JOIN property_addresses pa
  ON pa.property_id = p.id
WHERE TRIM(pa.plot_no) = '{safe_plot}'
  AND TRIM(pa.road_no) = '{safe_road}'
LIMIT 5;
""".strip()

        pra_rows = run_select(sql_pra_lookup) or []

        # Handle no match
        if not pra_rows:
            if matched_road != raw_road:
                extra = f" (road interpreted as '{matched_road}' from DB)."
            else:
                extra = ""

            answer = (
                f"I couldn't find any property for plot {plot} and road {road}.{extra}\n"
                "Please check if the numbers are correct and try again."
            )
            state["final_answer"] = answer
            state["sql_query"] = "-- NOTE_SUMMARY_FLOW: no PRA found"
            state["sql_rows"] = []
            state["note_pra"] = None
            state["note_pdf_path"] = None
            state["error"] = None
            state["geometry"] = None

            # Reset note wizard
            self.note_flow = {
                "active": False,
                "step": None,
                "plot": None,
                "road": None,
            }
            return state

        # Handle multiple matches
        if len(pra_rows) > 1:
            pras = [row.get("pra") for row in pra_rows if row.get("pra")]
            pras_list = ", ".join(pras) if pras else "N/A"
            answer = (
                f"There are multiple properties for plot {plot} and road {road}.\n"
                f"Matching PRAs: {pras_list}.\n"
                "Please specify the exact PRA you want the note summary for."
            )
            state["final_answer"] = answer
            state["sql_query"] = sql_pra_lookup
            state["sql_rows"] = pra_rows
            state["note_pra"] = None
            state["note_pdf_path"] = None
            state["error"] = None

            # Reset wizard so user can start again with a precise PRA or new note flow
            self.note_flow = {
                "active": False,
                "step": None,
                "plot": None,
                "road": None,
            }
            return state

        # Exactly one PRA found
        pra = pra_rows[0].get("pra")

        if not pra:
            answer = (
                f"I found one property for plot {plot} and road {road}, "
                "but it does not have a PRA stored. I can't generate the note summary."
            )
            state["final_answer"] = answer
            state["sql_query"] = sql_pra_lookup
            state["sql_rows"] = pra_rows
            state["note_pra"] = None
            state["note_pdf_path"] = None
            state["error"] = None

            self.note_flow = {
                "active": False,
                "step": None,
                "plot": None,
                "road": None,
            }
            return state

        # 2) Generate the note PDF using the PRA
        summary_text, pdf_path, _current_rows, _history_rows = generate_property_note_pdf(
            llm=self.llm,
            pra=pra,
        )

        # As per your requirement: do NOT show summary or path
        state["final_answer"] = "note summary saved;"
        state["sql_query"] = sql_pra_lookup + f"\n-- NOTE_SUMMARY for PRA {pra}"
        state["sql_rows"] = pra_rows
        state["note_pra"] = pra
        state["note_pdf_path"] = pdf_path
        state["error"] = None

        # Reset note wizard
        self.note_flow = {
            "active": False,
            "step": None,
            "plot": None,
            "road": None,
        }

        # ⚠️ Do NOT save history here
        return state

    def note_summary_direct(self, state: ChatbotState) -> ChatbotState:
        """
        Single-turn note summary:
        e.g. 'generate note summary of plot 30 road 14' or 'note summary for 30/14'.

        We reuse the same PRA lookup + PDF logic as collect_road,
        but parse plot+road from the sentence instead of using note_flow.
        """
        plot_raw, road_raw = parse_plot_road_from_text(state["user_query"])

        # If parsing fails for some reason, fall back to the interactive flow
        if not plot_raw or not road_raw:
            return self.start_note_summary(state)

        # Fuzzy match plot + road
        plot = self._fuzzy_match_column("plot_no", plot_raw.strip(), threshold=80)
        road = self._fuzzy_match_column("road_no", road_raw.strip(), threshold=65)

        safe_plot = plot.replace("'", "''")
        safe_road = road.replace("'", "''")

        sql_pra_lookup = f"""
SELECT p.pra
FROM properties p
JOIN property_addresses pa
  ON pa.property_id = p.id
WHERE TRIM(pa.plot_no) = '{safe_plot}'
  AND TRIM(pa.road_no) = '{safe_road}'
LIMIT 5;
""".strip()

        pra_rows = run_select(sql_pra_lookup) or []

        # No match
        if not pra_rows:
            state["final_answer"] = (
                f"I couldn't find any property for plot {plot} and road {road}.\n"
                "Please check if the numbers are correct and try again."
            )
            state["sql_query"] = "-- NOTE_SUMMARY_FLOW: no PRA found"
            state["sql_rows"] = []
            state["note_pra"] = None
            state["note_pdf_path"] = None
            state["error"] = None
            state["geometry"] = None
            return state

        # Multiple matches
        if len(pra_rows) > 1:
            pras = [row.get("pra") for row in pra_rows if row.get("pra")]
            pras_list = ", ".join(pras) if pras else "N/A"
            state["final_answer"] = (
                f"There are multiple properties for plot {plot} and road {road}.\n"
                f"Matching PRAs: {pras_list}.\n"
                "Please specify the exact PRA you want the note summary for."
            )
            state["sql_query"] = sql_pra_lookup
            state["sql_rows"] = pra_rows
            state["note_pra"] = None
            state["note_pdf_path"] = None
            state["error"] = None
            state["geometry"] = None
            return state

        # Exactly one PRA
        pra = pra_rows[0].get("pra")
        if not pra:
            state["final_answer"] = (
                f"I found one property for plot {plot} and road {road}, "
                "but it does not have a PRA stored. I can't generate the note summary."
            )
            state["sql_query"] = sql_pra_lookup
            state["sql_rows"] = pra_rows
            state["note_pra"] = None
            state["note_pdf_path"] = None
            state["error"] = None
            state["geometry"] = None
            return state

        # Generate the note PDF
        summary_text, pdf_path, _current_rows, _history_rows = generate_property_note_pdf(
            llm=self.llm,
            pra=pra,
        )

        state["final_answer"] = "note summary saved;"
        state["sql_query"] = sql_pra_lookup + f"\n-- NOTE_SUMMARY for PRA {pra}"
        state["sql_rows"] = pra_rows
        state["note_pra"] = pra
        state["note_pdf_path"] = pdf_path
        state["error"] = None
        state["geometry"] = None
        return state


    def map_lookup(self, state: ChatbotState) -> ChatbotState:
        """
        Handle: 'show the map of plot 30 road 15'.

        Steps:
        - parse plot + road from the question
        - fuzzy-match both against DB
        - resolve PRA
        - fetch geometry from pbchs_map
        - populate final_answer, sql_query, sql_rows, geometry
        """
        query = state["user_query"]

        plot_raw, road_raw = parse_plot_road_from_text(query)

        if not plot_raw or not road_raw:
            state["final_answer"] = (
                "To show the property map, please tell me both the plot and "
                "road number, e.g. 'show the map of plot 30 road 14'."
            )
            state["sql_query"] = "-- MAP_LOOKUP: could not parse plot/road"
            state["sql_rows"] = []
            state["geometry"] = None
            state["error"] = None
            return state

        # Fuzzy-match against canonical plot/road values in DB
        plot = self._fuzzy_match_column("plot_no", plot_raw, threshold=80)
        road = self._fuzzy_match_column("road_no", road_raw, threshold=65)

        # Resolve PRA
        pra_sql, pra_rows = lookup_pra_for_plot_road(plot, road)

        # No matching property at all
        if not pra_rows:
            state["final_answer"] = (
                f"I couldn't find any property for plot {plot} and road {road}, "
                "so I don't have a map to show."
            )
            state["sql_query"] = pra_sql
            state["sql_rows"] = []
            state["geometry"] = None
            state["error"] = None
            return state

        # More than one PRA for that plot/road
        if len(pra_rows) > 1:
            pras = [r.get("pra") for r in pra_rows if r.get("pra")]
            pras_list = ", ".join(pras) if pras else "N/A"

            state["final_answer"] = (
                f"There are multiple properties for plot {plot} and road {road}.\n"
                f"Matching PRAs: {pras_list}.\n"
                "Please specify exactly which PRA you want the map for, e.g. "
                "'show the map for PRA 23|18|Punjabi Bagh East'."
            )
            state["sql_query"] = pra_sql
            state["sql_rows"] = pra_rows
            state["geometry"] = None
            state["error"] = None
            return state

        # Exactly one PRA
        pra = pra_rows[0].get("pra")
        if not pra:
            state["final_answer"] = (
                f"I found one property for plot {plot} and road {road}, "
                "but it does not have a PRA stored, so I can't fetch a map."
            )
            state["sql_query"] = pra_sql
            state["sql_rows"] = pra_rows
            state["geometry"] = None
            state["error"] = None
            return state

        # Fetch geometry from pbchs_map
        map_sql, map_rows = fetch_map_for_pra(pra)

        if not map_rows:
            state["final_answer"] = (
                f"I found property {pra} for plot {plot} and road {road}, "
                "but there is currently no map geometry stored for it."
            )
            state["sql_query"] = map_sql
            state["sql_rows"] = []
            state["geometry"] = None
            state["error"] = None
            return state

        # Extract GeoJSON geometries for the full-stack team
        geometry: list[dict] = []
        for row in map_rows:
            feature = row.get("feature")
            if isinstance(feature, dict) and "geometry" in feature:
                geometry.append(feature["geometry"])

        state["final_answer"] = (
            f"Map geometry is available for property {pra} (plot {plot}, road {road}). "
            "I've returned the GeoJSON feature(s) that your frontend can render."
        )
        state["sql_query"] = map_sql
        state["sql_rows"] = map_rows
        state["geometry"] = geometry
        state["error"] = None
        return state


    def extract_entities(self, state: ChatbotState) -> ChatbotState:
        """Extract and enrich entities"""
        ner = extract_property_entities(state["user_query"], self.llm)
        ner = fuzzy_enrich_entities(ner)
        self.memory.update_from_entities(ner)
        state["ner_entities"] = ner
        return state
    
    def build_standalone(self, state: ChatbotState) -> ChatbotState:
        """Build standalone question"""
        standalone_info = build_standalone_question(
            llm=self.llm,
            raw_query=state["user_query"],
            history_messages=state["history_messages"],
            ner_entities=state["ner_entities"],
            focus_property=self.memory.focus_property,
            focus_person=self.memory.focus_person
        )

        # Post-process the standalone question (plot/road/person, Punjabi Bagh, 30/14 etc.)
        original_ner = state.get("ner_entities", {}) or {}
        standalone_q = standalone_info.get("standalone_question", "")

        processed_q, updated_ner = self._postprocess_standalone_question(
            standalone_q,
            dict(original_ner),  # shallow copy
        )

        standalone_info["standalone_question"] = processed_q
        state["ner_entities"] = updated_ner
        state["standalone_info"] = standalone_info
        return state

    

    def retrieve_context(self, state: ChatbotState) -> ChatbotState:
        """Retrieve SQL examples and schema from vector store"""
        standalone_question = state["standalone_info"]["standalone_question"]
        
        # Get SQL examples
        sql_matches = self.vstore.query_sql_examples(standalone_question, top_k=5)
        best_sim = max((m["similarity"] for m in sql_matches), default=0.0)
        
        if best_sim >= SQL_SIMILARITY_THRESHOLD:
            state["sql_matches"] = sql_matches[:3]
        else:
            state["sql_matches"] = []
        
        # Get schema docs
        state["schema_matches"] = self.vstore.query_schema(standalone_question, top_k=5)
        return state
        
    def generate_sql_node(self, state: ChatbotState) -> ChatbotState:
        """Generate SQL query"""
        # 1) First draft from the SQL LLM
        original_sql = generate_sql(
            llm=self.llm,
            standalone_question=state["standalone_info"]["standalone_question"],
            ner_entities=state["ner_entities"],
            schema_matches=state["schema_matches"],
            sql_example_matches=state["sql_matches"],
        )

        # 2) Try the normal validator first (old behaviour)
        try:
            validation_result = validate_and_maybe_regenerate_sql(
                llm=self.llm,
                sql_query=original_sql,
                question=state["standalone_info"]["standalone_question"],
                max_retries=1,
            )
            state["sql_query"] = validation_result["sql"]
            state["error"] = None
            return state

        except Exception as e:
            # 3) If the validator itself fails (e.g. forbidden column, table, etc.),
            #    call the dynamic repairer once using that error message.
            validation_err = str(e)

            repair_prompt = f"""
    You previously wrote this PostgreSQL SELECT query:

    {original_sql}

    It FAILED validation with this error:

    {validation_err}

    Rewrite the query so that it:
    - Still answers the same question.
    - Uses ONLY allowed tables/columns from the provided schema.
    - Does NOT use any forbidden or non-existent columns.
    - Respects all JSON rules:
    * ownership_records.buyer_portion is JSON. Do NOT group by or compare the raw JSON.
        If needed, use (ownership_records.buyer_portion->>0) or
        CAST(ownership_records.buyer_portion->>0 AS numeric).
    * sale_deeds.signing_date is JSON/text. To get a DATE use:
        to_date(alias.signing_date->>0, 'DD/MM/YYYY')
        where alias is the table alias (e.g. sd).
    * NEVER GROUP BY a raw JSON column; only group by text/number/date expressions.

    Return ONLY a single PostgreSQL SELECT statement ending with a semicolon.
    """.strip()

            repaired_sql = self.llm.generate_text(
                system_prompt=SQL_GENERATION_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                max_tokens=400,
                temperature=0.0,
            )

            # 4) Try to validate the repaired SQL once more
            try:
                validation_result = validate_and_maybe_regenerate_sql(
                    llm=self.llm,
                    sql_query=repaired_sql,
                    question=state["standalone_info"]["standalone_question"],
                    max_retries=1,
                )
                state["sql_query"] = validation_result["sql"]
                state["error"] = None
            except Exception as e2:
                # If this ALSO fails, we stop and surface the error
                state["sql_query"] = f"-- ERROR: unsafe SQL blocked: {e2}"
                state["error"] = str(e2)

            return state

    def execute_sql(self, state: ChatbotState) -> ChatbotState:
        """Execute SQL query"""
        if state.get("error"):
            state["sql_rows"] = []
            return state

        try:
            # Try running the (already validated) query
            sql_rows = run_select(state["sql_query"])
            state["sql_rows"] = sql_rows
            state["error"] = None

        except ProgrammingError as e:
            # For ANY PostgreSQL programming error, try a single LLM-based repair.
            db_err = str(getattr(e, "orig", e))

            repair_prompt = f"""
    You previously wrote this PostgreSQL SELECT query:

    {state["sql_query"]}

    It FAILED with the following PostgreSQL error:

    {db_err}

    You must now rewrite the query so that it runs successfully against the same schema
    and answers the same user question.

    Important constraints:

    - Only produce a single PostgreSQL SELECT statement ending with a semicolon.
    - Do NOT modify data: no INSERT, UPDATE, DELETE, ALTER, DROP, TRUNCATE, GRANT, REVOKE.
    - Do NOT use any tables or columns that are not in the provided schema.

    - If any columns are JSON (for example ownership_records.buyer_portion or sale_deeds.signing_date),
    do NOT group by or compare the raw JSON value directly.
    Instead, use a text/number/date expression such as:
        - (ownership_records.buyer_portion->>0)
        - (sale_deeds.signing_date->>0)
        - to_date(sale_deeds.signing_date->>0, 'DD/MM/YYYY')
    - NEVER GROUP BY a raw JSON column. If grouping is required, group by a TEXT/NUMERIC/DATE expression.
    - Never use persons.dob as a transaction or ownership-change date.

    Return ONLY the corrected PostgreSQL SELECT query with a semicolon.
    """.strip()

            repaired_sql = self.llm.generate_text(
                system_prompt=SQL_GENERATION_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                max_tokens=400,
                temperature=0.0,
            )

            # Validate repaired SQL and try once more
            try:
                validation_result = validate_and_maybe_regenerate_sql(
                    llm=self.llm,
                    sql_query=repaired_sql,
                    question=state["standalone_info"]["standalone_question"],
                    max_retries=1,
                )
                state["sql_query"] = validation_result["sql"]
                sql_rows = run_select(state["sql_query"])
                state["sql_rows"] = sql_rows
                state["error"] = None

            except Exception as e2:
                state["sql_rows"] = []
                state["sql_query"] = f"-- ERROR after repair: {e2}"
                state["error"] = str(e2)

        except Exception as e:
            state["sql_rows"] = []
            state["sql_query"] = f"-- ERROR: {e}"
            state["error"] = str(e)

        return state

    def build_answer(self, state: ChatbotState) -> ChatbotState:
        """Build final answer"""
        final_answer = build_final_answer(
            llm=self.llm,
            user_query=state["user_query"],
            standalone_question=state["standalone_info"]["standalone_question"],
            sql_query=state["sql_query"],
            sql_rows=state["sql_rows"],
            history_messages=state["history_messages"],
            on_token=self.on_token,
        )
        state["final_answer"] = final_answer
        state["geometry"] = None
        return state

    
    def save_history(self, state: ChatbotState) -> ChatbotState:
        """Save conversation to history in Mongo."""
        label = state["classification"].get("label")

        # For property_talk, store the standalone question in history;
        # for others, keep the original user_query.
        if label == "property_talk" and state.get("standalone_info"):
            stored_user_message = state["standalone_info"].get(
                "standalone_question",
                state["user_query"],
            )
        else:
            stored_user_message = state["user_query"]

        if state.get("sql_rows") or label != "property_talk":
            self.history.add_exchange(
                user_id=self.user_id,
                user_message=stored_user_message,
                assistant_message=state["final_answer"],
                thread_id=self.thread_id,
            )
        return state

    
    @traceable(run_type="chain", name="property_chatbot_graph")
    def run(
        self,
        user_query: str,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, str, list[dict], Optional[list[dict]]]:

        """Run the graph"""

        # Install streaming callback for this run
        self.on_token = on_token

        initial_state: ChatbotState = {
            # Input
            "user_query": user_query,

            # Intermediate results
            "history_messages": [],
            "classification": {},
            "ner_entities": {},
            "standalone_info": {},
            "sql_matches": [],
            "schema_matches": [],
            "sql_query": "",
            "sql_rows": [],

            # Output
            "final_answer": "",
            "error": None,

            # Note-summary extras
            "note_pra": None,
            "note_pdf_path": None,

            # Map extras
            "geometry": None,
        }

        final_state = self.graph.invoke(initial_state)

        # Clear callback so it doesn't leak into the next run
        self.on_token = None

        return (
            final_state["final_answer"],
            final_state["sql_query"],
            final_state["sql_rows"],
            final_state.get("geometry"),
        )

