"""Pure retriever: Ollama embeddings + in-memory numpy cosine similarity.

No vector database is used; this is the most transparent implementation and
shows exactly what a vector store does under the hood.
"""

from __future__ import annotations

import numpy as np

from localagent.config import get_settings
from localagent.llm import build_embeddings
from localagent.rag.base import Document, RetrievedChunk, Retriever, chunk_text


class NumpyRetriever(Retriever):
    """Brute-force cosine-similarity search over an in-memory embedding matrix."""

    name = "pure-numpy"

    def __init__(self) -> None:
        self._embeddings = build_embeddings()
        self._matrix: np.ndarray | None = None
        self._chunks: list[str] = []
        self._sources: list[str] = []

    def index(self, documents: list[Document]) -> None:
        for doc in documents:
            for chunk in chunk_text(doc.text):
                self._chunks.append(chunk)
                self._sources.append(doc.source)
        if not self._chunks:
            return
        vectors = np.array(self._embeddings.embed_documents(self._chunks), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._matrix = vectors / np.clip(norms, 1e-12, None)

    def known_sources(self) -> set[str]:
        return set(self._sources)

    def add_document(self, document: Document) -> int:
        chunks = chunk_text(document.text)
        if not chunks:
            return 0
        vectors = np.array(self._embeddings.embed_documents(chunks), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.clip(norms, 1e-12, None)
        self._matrix = normalized if self._matrix is None else np.vstack([self._matrix, normalized])
        self._chunks.extend(chunks)
        self._sources.extend([document.source] * len(chunks))
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if self._matrix is None:
            return []
        top_k = top_k or get_settings().rag_top_k
        query_vec = np.array(self._embeddings.embed_query(query), dtype=np.float32)
        query_vec /= max(float(np.linalg.norm(query_vec)), 1e-12)
        scores = self._matrix @ query_vec
        order = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(text=self._chunks[i], source=self._sources[i], score=float(scores[i]))
            for i in order
        ]
