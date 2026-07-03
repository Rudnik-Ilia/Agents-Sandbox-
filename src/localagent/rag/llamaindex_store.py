"""Retriever implemented with LlamaIndex's VectorStoreIndex over Ollama models."""

from __future__ import annotations

from llama_index.core import Document as LIDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding

from localagent.config import get_settings
from localagent.rag.base import Document, RetrievedChunk, Retriever


class LlamaIndexRetriever(Retriever):
    """Wrap a LlamaIndex `VectorStoreIndex` retriever behind the common interface."""

    name = "llamaindex"

    def __init__(self) -> None:
        settings = get_settings()
        self._embed_model = OllamaEmbedding(
            model_name=settings.embed_model,
            base_url=settings.ollama_base_url,
        )
        self._splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        self._index: VectorStoreIndex | None = None
        self._retriever = None
        self._sources: set[str] = set()

    def index(self, documents: list[Document]) -> None:
        li_docs = [LIDocument(text=doc.text, metadata={"source": doc.source}) for doc in documents]
        self._index = VectorStoreIndex.from_documents(
            li_docs,
            embed_model=self._embed_model,
            transformations=[self._splitter],
        )
        self._sources = {doc.source for doc in documents}
        self._retriever = self._index.as_retriever(similarity_top_k=get_settings().rag_top_k)

    def known_sources(self) -> set[str]:
        return set(self._sources)

    def add_document(self, document: Document) -> int:
        li_doc = LIDocument(text=document.text, metadata={"source": document.source})
        if self._index is None:
            self.index([document])
            return len(self._splitter.get_nodes_from_documents([li_doc]))
        nodes = self._splitter.get_nodes_from_documents([li_doc])
        self._index.insert_nodes(nodes)
        self._sources.add(document.source)
        self._retriever = self._index.as_retriever(similarity_top_k=get_settings().rag_top_k)
        return len(nodes)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if self._retriever is None:
            return []
        nodes = self._retriever.retrieve(query)
        results: list[RetrievedChunk] = []
        for node in nodes[: (top_k or get_settings().rag_top_k)]:
            results.append(
                RetrievedChunk(
                    text=node.get_content(),
                    source=str(node.metadata.get("source", "?")),
                    score=float(node.score or 0.0),
                )
            )
        return results
