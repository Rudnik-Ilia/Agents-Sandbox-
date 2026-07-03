"""Pure-stack retriever using Chroma as a persistent local vector database.

Embeddings are produced by Ollama (`mxbai-embed-large`) and supplied to Chroma
explicitly, so the only role Chroma plays is durable storage and ANN search.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from localagent.config import PROJECT_ROOT, get_settings
from localagent.llm import build_embeddings
from localagent.rag.base import Document, RetrievedChunk, Retriever, chunk_text

_COLLECTION = "corpus"


class ChromaRetriever(Retriever):
    """Cosine-similarity search backed by a persistent Chroma collection."""

    name = "pure-chroma"

    def __init__(self) -> None:
        self._embeddings = build_embeddings()
        self._client = chromadb.PersistentClient(
            path=str(PROJECT_ROOT / ".chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            _COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def is_populated(self) -> bool:
        return self._collection.count() > 0

    def reset(self) -> None:
        if _has_collection(self._client):
            self._client.delete_collection(_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            _COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def index(self, documents: list[Document]) -> None:
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, str]] = []
        for doc in documents:
            for i, chunk in enumerate(chunk_text(doc.text)):
                ids.append(f"{doc.source}:{i}")
                texts.append(chunk)
                metadatas.append({"source": doc.source})
        if not texts:
            return
        vectors = self._embeddings.embed_documents(texts)
        self._collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)

    def known_sources(self) -> set[str]:
        data = self._collection.get(include=["metadatas"])
        return {str(m.get("source")) for m in data.get("metadatas", []) if m}

    def add_document(self, document: Document) -> int:
        chunks = chunk_text(document.text)
        if not chunks:
            return 0
        offset = self._collection.count()
        ids = [f"{document.source}:{offset + i}" for i in range(len(chunks))]
        metadatas = [{"source": document.source} for _ in chunks]
        vectors = self._embeddings.embed_documents(chunks)
        self._collection.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or get_settings().rag_top_k
        query_vec = self._embeddings.embed_query(query)
        result = self._collection.query(query_embeddings=[query_vec], n_results=top_k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: list[RetrievedChunk] = []
        for text, meta, distance in zip(documents, metadatas, distances, strict=False):
            chunks.append(
                RetrievedChunk(text=text, source=str(meta.get("source", "?")), score=1.0 - float(distance))
            )
        return chunks


def _has_collection(client: chromadb.ClientAPI) -> bool:
    return any(c.name == _COLLECTION for c in client.list_collections())
