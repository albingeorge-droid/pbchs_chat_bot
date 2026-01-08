from __future__ import annotations
import json
import re
from typing import List, Dict, Any

from openai import OpenAI

from config import settings



class GroqClient:
    """
    Thin wrapper around the OpenAI Chat Completions API for:
    - text / JSON generation via chat completions
    (Embeddings are handled separately by SentenceEmbeddingClient.)
    """

    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in .env.local")

        # OpenAI Python client
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.chat_model = settings.openai_chat_model


    # ------------------------------------------------------------------
    # Core LLM call (this is what LangSmith traces)
    # ------------------------------------------------------------------


    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8000,
        temperature: float = 0.8,
    ):
        """
        Low-level Groq call that is actually traced by LangSmith.
        Returns the raw ChatCompletion object.
        """
        return self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Public helpers used by the rest of the code
    # ------------------------------------------------------------------

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 8000,
        temperature: float = 0.8,
    ) -> str:
        """
        High-level helper: returns plain assistant text.
        This wraps `_chat_completion` so LangSmith still sees the full trace.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = self._chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        return text.strip()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 8000,
        temperature: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Ask Groq to return ONLY a JSON object; parse safely.
        Reuses `generate_text`, so the trace & tokens are logged via `_chat_completion`.
        """
        text = self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        raw = (text or "").strip()

        # ✅ remove markdown code fences like ```json ... ``` or ``` ... ```
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        # ✅ try direct JSON parse
        try:
            return json.loads(raw)
        except Exception:
            # ✅ fallback: extract first {...} block if model added extra text
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {}

