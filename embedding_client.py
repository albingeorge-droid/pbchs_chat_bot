from __future__ import annotations
from typing import List

from sentence_transformers import SentenceTransformer

from config import settings


class SentenceEmbeddingClient:
    """
    Wrapper around SentenceTransformers for embeddings.
    Used by the vector store for:
    - schema docs
    - SQL example docs
    - query embeddings
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.sentence_model_name
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",  # kept for API compatibility, not used
    ) -> List[List[float]]:
        if not texts:
            return []
        # normalize_embeddings=True gives cosine similarity-friendly vectors
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=False,
            normalize_embeddings=True,
        )
        # Convert to plain Python lists for Chroma
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
