"""Retriever using LlamaIndex orchestration over a persistent Chroma vector store.

Contrast with the two pure-Chroma/pure-numpy backends: here LlamaIndex owns the
pipeline (splitting, embedding, indexing, retrieval) while Chroma is just the
external, persistent vector store - the common production layering.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import Document as LIDocument
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from localagent.config import PROJECT_ROOT, get_settings
from localagent.rag.base import Document, RetrievedChunk, Retriever

_COLLECTION = "li_corpus"


class LlamaIndexChromaRetriever(Retriever):
    """LlamaIndex `VectorStoreIndex` backed by a persistent Chroma collection."""

    name = "llamaindex-chroma"

    def __init__(self, namespace: str | None = None) -> None:
        settings = get_settings()
        self._embed_model = OllamaEmbedding(model_name=settings.embed_model, base_url=settings.ollama_base_url)
        self._splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        base_path = PROJECT_ROOT / ".chroma_llamaindex"
        path = base_path / namespace if namespace else base_path
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(_COLLECTION)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._index: VectorStoreIndex | None = None

    def _get_index(self) -> VectorStoreIndex:
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(
                self._vector_store, embed_model=self._embed_model
            )
        return self._index

    def is_populated(self) -> bool:
        return self._collection.count() > 0

    def reset(self) -> None:
        if any(c.name == _COLLECTION for c in self._client.list_collections()):
            self._client.delete_collection(_COLLECTION)
        self._collection = self._client.get_or_create_collection(_COLLECTION)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._index = None

    def index(self, documents: list[Document]) -> None:
        li_docs = [LIDocument(text=doc.text, metadata={"source": doc.source}) for doc in documents]
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex.from_documents(
            li_docs,
            storage_context=storage_context,
            embed_model=self._embed_model,
            transformations=[self._splitter],
        )

    def known_sources(self) -> set[str]:
        data = self._collection.get(include=["metadatas"])
        return {str(m.get("source")) for m in data.get("metadatas", []) if m and m.get("source")}

    def add_document(self, document: Document) -> int:
        li_doc = LIDocument(text=document.text, metadata={"source": document.source})
        nodes = self._splitter.get_nodes_from_documents([li_doc])
        self._get_index().insert_nodes(nodes)
        return len(nodes)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or get_settings().rag_top_k
        retriever = self._get_index().as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        return [
            RetrievedChunk(
                text=node.get_content(),
                source=str(node.metadata.get("source", "?")),
                score=float(node.score or 0.0),
            )
            for node in nodes[:top_k]
        ]
