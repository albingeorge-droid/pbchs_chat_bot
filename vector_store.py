from __future__ import annotations
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb import Documents, Metadatas, IDs

from langsmith import traceable

from config import settings
from embedding_client import SentenceEmbeddingClient
from prompts import TABLE_SCHEMAS, SQL_EXAMPLES


class PropertyVectorStore:
    """
    Wrapper over Chroma to store:
    - Schema docs (kind='schema')
    - SQL examples (kind='sql_example')
    """

    def __init__(self, embedder: SentenceEmbeddingClient | None = None):
        # Persistent client on disk
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection_name = settings.chroma_collection_name

        # We do NOT use an embedding_function here; we pass embeddings explicitly
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # <= use cosine distance
        )


        self.embedder = embedder or SentenceEmbeddingClient()

        # 🔍 Auto-bootstrap vector DB if empty
        try:
            count = self.collection.count()
        except Exception:
            # If count() is not available / blows up, treat as empty
            count = 0

        if count == 0:
            print(
                f"[VectorStore] No vectors found in collection "
                f"'{self.collection_name}' — rebuilding index..."
            )
            self.rebuild_index()

    # ---------- Bootstrap / upsert ----------

    def _build_schema_docs(self) -> Tuple[Documents, IDs, Metadatas]:
        """
        Build Chroma docs for table schemas from TABLE_SCHEMAS.

        Expected TABLE_SCHEMAS format (from prompts.py):
        [
            {
                "table": "properties",
                "description": "...",
            },
            ...
        ]
        """
        docs: Documents = []
        ids: IDs = []
        metas: Metadatas = []

        for i, item in enumerate(TABLE_SCHEMAS):
            table_name = item["table"]
            doc_id = f"schema-{table_name}"
            text = f"Table {table_name} description:\n{item['description']}"
            docs.append(text)
            ids.append(doc_id)
            metas.append(
                {
                    "kind": "schema",
                    "table": table_name,
                }
            )
        return docs, ids, metas

    def _build_sql_example_docs(self) -> Tuple[Documents, IDs, Metadatas]:
        """
        Build Chroma docs for SQL examples from SQL_EXAMPLES.

        IMPORTANT:
        - The *document* we embed is ONLY the natural-language question.
        - The SQL itself is stored in metadata (so we can still use it later).
        """
        docs: Documents = []
        ids: IDs = []
        metas: Metadatas = []

        for ex in SQL_EXAMPLES:
            doc_id = f"sql-example-{ex['id']}"

            # 👇 Only the question is embedded
            text = ex["question"]

            docs.append(text)
            ids.append(doc_id)
            metas.append(
                {
                    "kind": "sql_example",
                    "example_id": ex["id"],
                    "tables": ",".join(ex["tables"]),
                    "question": ex["question"],
                    "sql": ex["sql"],  # 👈 keep the full SQL in metadata
                }
            )
        return docs, ids, metas


    @traceable(run_type="chain", name="rebuild_index")
    def rebuild_index(self):
        """
        Wipe and rebuild the Chroma collection for:
        - table schemas
        - SQL examples
        """
        print(
            f"[VectorStore] Dropping existing collection '{self.collection_name}' (if it exists)..."
        )
        try:
            # Safe even if collection doesn't exist
            self.client.delete_collection(name=self.collection_name)
        except Exception as e:
            # Not fatal
            print(f"[VectorStore] delete_collection warning: {e}")

        # Re-create empty collection
        # Re-create empty collection with cosine metric
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # <= use cosine distance
        )


        print("[VectorStore] Rebuilding index with schema + SQL examples...")
        # 1) Schema docs
        schema_docs, schema_ids, schema_metas = self._build_schema_docs()
        if schema_docs:
            schema_embs = self.embedder.embed_texts(
                list(schema_docs), task_type="RETRIEVAL_DOCUMENT"
            )
            self.collection.add(
                documents=schema_docs,
                embeddings=schema_embs,
                metadatas=schema_metas,
                ids=schema_ids,
            )

        # 2) SQL examples
        ex_docs, ex_ids, ex_metas = self._build_sql_example_docs()
        if ex_docs:
            ex_embs = self.embedder.embed_texts(
                list(ex_docs), task_type="RETRIEVAL_DOCUMENT"
            )
            self.collection.add(
                documents=ex_docs,
                embeddings=ex_embs,
                metadatas=ex_metas,
                ids=ex_ids,
            )


        total = len(schema_ids) + len(ex_ids)
        print(f"[VectorStore] Loaded {total} items into Chroma.")

        # 👇 This shows up in LangSmith as the run Output
        return {
            "schema_docs": len(schema_ids),
            "sql_example_docs": len(ex_ids),
            "total_items": total,
        }
    # ---------- Query helpers ----------

    @traceable(run_type="retriever", name="query_sql_examples")
    def query_sql_examples(
        self, question: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k SQL examples for the given question.
        """
        q_emb = self.embedder.embed_query(question)

        result = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={"kind": "sql_example"},
            include=["documents", "metadatas", "distances"],
        )

        matches: List[Dict[str, Any]] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            similarity = 1.0 - float(dist)
            matches.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "similarity": similarity,
                }
            )
        return matches

    @traceable(run_type="retriever", name="query_schema")
    def query_schema(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k schema docs relevant to the question.
        """
        q_emb = self.embedder.embed_query(question)
        result = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={"kind": "schema"},
            include=["documents", "metadatas", "distances"],
        )

        matches: List[Dict[str, Any]] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            similarity = 1.0 - float(dist)
            matches.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "similarity": similarity,
                }
            )
        return matches


# ---------------------------------------------------------------------
# Script entrypoint for embedding schema + SQL examples once.
# Run this with:  python -m utils.chat_help.vector_store
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Bootstrapping Chroma with schema + SQL examples using sentence-transformers embeddings...")
    embedder = SentenceEmbeddingClient()
    store = PropertyVectorStore(embedder)
    store.rebuild_index()
    print("✅ Chroma index rebuilt.")

# python -m utils.chat_help_langGraph_Openai.vector_store
