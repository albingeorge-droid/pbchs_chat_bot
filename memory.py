from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Dict, Any

from pymongo import MongoClient

from config import settings


class HistoryManager:
    """
    Store chat history in MongoDB as individual message documents.

    Each document looks like:
      {
        "user_id": "...",
        "thread_id": "...",   # can be same as user_id
        "role": "user" | "assistant",
        "content": "...",
        "created_at": datetime(...)
      }
    """

    def __init__(self, client: MongoClient | None = None):
        self.client = client or MongoClient(settings.mongo_uri)
        self.db = self.client[settings.mongo_db]
        self.collection = self.db[settings.mongo_history_collection]
        self.max_docs_per_thread = 20  # ✅ keep only last 20 messages per user+thread

    # --------- read last k messages for a given user / thread ---------

    def last_messages(
        self,
        user_id: str,
        k: int = 6,
        thread_id: str | None = None,
    ) -> List[Dict[str, str]]:
        """
        Return last k messages as:
        [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."},
          ...
        ]
        """
        query: Dict[str, Any] = {"user_id": user_id}
        if thread_id:
            query["thread_id"] = thread_id

        cursor = (
            self.collection.find(query)
            .sort("created_at", -1)   # newest first
            .limit(k)
        )
        docs = list(cursor)
        docs.reverse()  # oldest -> newest

        return [
            {"role": d["role"], "content": d.get("content", "")}
            for d in docs
        ]

    # --------- append one user/assistant exchange ---------

    def add_exchange(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        thread_id: str | None = None,
    ) -> None:
        """
        Store the user + assistant messages as two docs.
        """
        thread_id = thread_id or user_id
        now = datetime.now(timezone.utc)

        docs = [
            {
                "user_id": user_id,
                "thread_id": thread_id,
                "role": "user",
                "content": user_message,
                "created_at": now,
            },
            {
                "user_id": user_id,
                "thread_id": thread_id,
                "role": "assistant",
                "content": assistant_message,
                "created_at": now,
            },
        ]
        self.collection.insert_many(docs)

        # ✅ PRUNE: keep only the latest `max_docs_per_thread` docs for this user+thread
        max_docs = self.max_docs_per_thread

        # Find all docs for this user+thread, newest first, skip the newest `max_docs`
        cursor = (
            self.collection.find(
                {"user_id": user_id, "thread_id": thread_id},
                {"_id": 1},                       # only need _id
            )
            .sort("created_at", -1)               # newest → oldest
            .skip(max_docs)                       # everything after the newest N
        )

        old_ids = [doc["_id"] for doc in cursor]
        if old_ids:
            self.collection.delete_many({"_id": {"$in": old_ids}})



class ConversationMemory:
    """
    Lightweight in-memory focus tracking:
    - last property (PRA / file_name)
    - last person name
    - last SQL rows
    """

    def __init__(self) -> None:
        self.focus_property: Dict[str, Any] | None = None
        self.focus_person: str | None = None
        self.last_sql_rows: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.focus_property = None
        self.focus_person = None
        self.last_sql_rows = []

    # ---- called from NER step ----
    def update_from_entities(self, entities: Dict[str, Any] | None) -> None:
        entities = entities or {}

        # property focus
        if any(
            entities.get(k)
            for k in ["pra", "file_no", "file_name", "plot_no", "road_no", "area"]
        ):
            self.focus_property = {
                "pra": entities.get("pra")
                or (self.focus_property or {}).get("pra"),
                "file_name": entities.get("file_name")
                or (self.focus_property or {}).get("file_name"),
            }

        # person focus
        for key in ["person_name", "owner_name", "name"]:
            val = entities.get(key)
            if isinstance(val, str) and val.strip():
                self.focus_person = val.strip()
                break

    # ---- called after SQL execution ----
    def update_from_sql_rows(self, rows: List[Dict[str, Any]]):
        self.last_sql_rows = rows or []
        if not rows:
            return

        row = rows[0]
        if "pra" in row or "file_name" in row:
            self.focus_property = {
                "pra": row.get("pra"),
                "file_name": row.get("file_name"),
            }
        if "name" in row and isinstance(row["name"], str):
            self.focus_person = row["name"].strip()
