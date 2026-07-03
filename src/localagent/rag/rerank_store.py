"""Production-style retriever: hybrid (BM25 + dense) retrieval fused, then reranked.

Pipeline per query:
1. Dense retrieval - cosine similarity over Ollama embeddings.
2. Sparse retrieval - BM25 keyword scoring.
3. Fuse both rankings with Reciprocal Rank Fusion (RRF) into a candidate set.
4. Rerank the candidates with a cross-encoder and keep the top-k.

This mirrors a common production retrieval stack on a small, local scale.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from localagent.config import get_settings
from localagent.llm import build_embeddings
from localagent.rag.base import Document, RetrievedChunk, Retriever, chunk_text


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _rrf(dense: np.ndarray, sparse: np.ndarray, top_n: int, k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion of two score arrays; returns candidate chunk indices."""
    dense_rank = np.empty(len(dense), dtype=int)
    dense_rank[np.argsort(dense)[::-1]] = np.arange(len(dense))
    sparse_rank = np.empty(len(sparse), dtype=int)
    sparse_rank[np.argsort(sparse)[::-1]] = np.arange(len(sparse))
    fused = 1.0 / (k + dense_rank) + 1.0 / (k + sparse_rank)
    return list(np.argsort(fused)[::-1][:top_n])


class RerankRetriever(Retriever):
    """Hybrid BM25 + dense retrieval with cross-encoder reranking."""

    name = "rerank"

    def __init__(self) -> None:
        self._embeddings = build_embeddings()
        self._chunks: list[str] = []
        self._sources: list[str] = []
        self._matrix: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None
        self._reranker = None

    def _ensure_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(get_settings().rerank_model)
        return self._reranker

    def _rebuild(self) -> None:
        vectors = np.array(self._embeddings.embed_documents(self._chunks), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._matrix = vectors / np.clip(norms, 1e-12, None)
        self._bm25 = BM25Okapi([_tokenize(c) for c in self._chunks])

    def index(self, documents: list[Document]) -> None:
        for doc in documents:
            for chunk in chunk_text(doc.text):
                self._chunks.append(chunk)
                self._sources.append(doc.source)
        if self._chunks:
            self._rebuild()

    def known_sources(self) -> set[str]:
        return set(self._sources)

    def add_document(self, document: Document) -> int:
        chunks = chunk_text(document.text)
        if not chunks:
            return 0
        self._chunks.extend(chunks)
        self._sources.extend([document.source] * len(chunks))
        self._rebuild()
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if self._matrix is None or self._bm25 is None:
            return []
        settings = get_settings()
        top_k = top_k or settings.rag_top_k

        query_vec = np.array(self._embeddings.embed_query(query), dtype=np.float32)
        query_vec /= max(float(np.linalg.norm(query_vec)), 1e-12)
        dense_scores = self._matrix @ query_vec
        sparse_scores = np.array(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)

        candidates = _rrf(dense_scores, sparse_scores, top_n=min(settings.fusion_candidates, len(self._chunks)))

        reranker = self._ensure_reranker()
        pairs = [[query, self._chunks[i]] for i in candidates]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, rerank_scores, strict=False), key=lambda pair: pair[1], reverse=True)

        return [
            RetrievedChunk(text=self._chunks[i], source=self._sources[i], score=float(score))
            for i, score in ranked[:top_k]
        ]
