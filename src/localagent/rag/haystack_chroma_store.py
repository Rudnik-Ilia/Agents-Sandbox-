"""Retriever using Haystack orchestration over a persistent Chroma vector store.

Haystack owns the pipeline (splitting, embedding, retrieval) while Chroma is the
external, persistent store - the same production layering as the LlamaIndex+Chroma
backend, but built from Haystack components.
"""

from __future__ import annotations

from haystack import Document as HSDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from localagent.config import PROJECT_ROOT, get_settings
from localagent.rag.base import Document, RetrievedChunk, Retriever

_COLLECTION = "hs_corpus"


class HaystackChromaRetriever(Retriever):
    """Haystack components backed by a persistent Chroma document store."""

    name = "haystack-chroma"

    def __init__(self, namespace: str | None = None) -> None:
        settings = get_settings()
        base_path = PROJECT_ROOT / ".chroma_haystack"
        path = base_path / namespace if namespace else base_path
        self._store = ChromaDocumentStore(
            collection_name=_COLLECTION,
            persist_path=str(path),
            distance_function="cosine",
        )
        self._splitter = DocumentSplitter(split_by="word", split_length=180, split_overlap=30)
        self._doc_embedder = OllamaDocumentEmbedder(model=settings.embed_model, url=settings.ollama_base_url)
        self._text_embedder = OllamaTextEmbedder(model=settings.embed_model, url=settings.ollama_base_url)
        self._retriever = ChromaEmbeddingRetriever(document_store=self._store)
        for component in (self._splitter, self._doc_embedder, self._text_embedder):
            if hasattr(component, "warm_up"):
                component.warm_up()

    def _embed_chunks(self, documents: list[HSDocument]) -> list[HSDocument]:
        chunks = self._splitter.run(documents=documents)["documents"]
        for chunk in chunks:
            chunk.meta = {
                key: value
                for key, value in (chunk.meta or {}).items()
                if isinstance(value, (str, int, float, bool))
            }
        return self._doc_embedder.run(documents=chunks)["documents"]

    def is_populated(self) -> bool:
        return self._store.count_documents() > 0

    def reset(self) -> None:
        self._store.delete_all_documents()

    def index(self, documents: list[Document]) -> None:
        hs_docs = [HSDocument(content=doc.text, meta={"source": doc.source}) for doc in documents]
        embedded = self._embed_chunks(hs_docs)
        self._store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)

    def known_sources(self) -> set[str]:
        return {str((doc.meta or {}).get("source")) for doc in self._store.filter_documents()}

    def add_document(self, document: Document) -> int:
        embedded = self._embed_chunks([HSDocument(content=document.text, meta={"source": document.source})])
        self._store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
        return len(embedded)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or get_settings().rag_top_k
        query_embedding = self._text_embedder.run(text=query)["embedding"]
        documents = self._retriever.run(query_embedding=query_embedding, top_k=top_k)["documents"]
        return [
            RetrievedChunk(
                text=doc.content or "",
                source=str((doc.meta or {}).get("source", "?")),
                score=float(doc.score or 0.0),
            )
            for doc in documents
        ]
