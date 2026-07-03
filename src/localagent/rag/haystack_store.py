"""Retriever implemented with Haystack components over Ollama embeddings."""

from __future__ import annotations

from haystack import Document as HSDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder

from localagent.config import get_settings
from localagent.rag.base import Document, RetrievedChunk, Retriever


class HaystackRetriever(Retriever):
    """Index documents in a Haystack `InMemoryDocumentStore` and query by embedding."""

    name = "haystack"

    def __init__(self) -> None:
        settings = get_settings()
        self._store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        self._splitter = DocumentSplitter(split_by="word", split_length=180, split_overlap=30)
        self._doc_embedder = OllamaDocumentEmbedder(model=settings.embed_model, url=settings.ollama_base_url)
        self._text_embedder = OllamaTextEmbedder(model=settings.embed_model, url=settings.ollama_base_url)
        self._retriever = InMemoryEmbeddingRetriever(document_store=self._store)
        for component in (self._splitter, self._doc_embedder, self._text_embedder):
            if hasattr(component, "warm_up"):
                component.warm_up()

    def index(self, documents: list[Document]) -> None:
        hs_docs = [HSDocument(content=doc.text, meta={"source": doc.source}) for doc in documents]
        chunks = self._splitter.run(documents=hs_docs)["documents"]
        embedded = self._doc_embedder.run(documents=chunks)["documents"]
        self._store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)

    def known_sources(self) -> set[str]:
        return {str((doc.meta or {}).get("source")) for doc in self._store.filter_documents()}

    def add_document(self, document: Document) -> int:
        hs_doc = HSDocument(content=document.text, meta={"source": document.source})
        chunks = self._splitter.run(documents=[hs_doc])["documents"]
        embedded = self._doc_embedder.run(documents=chunks)["documents"]
        self._store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
        return len(chunks)

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
