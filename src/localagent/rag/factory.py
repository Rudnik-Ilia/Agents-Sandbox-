"""Build and warm up a retriever backend by name."""

from __future__ import annotations

from localagent.logging_setup import AgentLogger
from localagent.rag.base import Retriever, load_corpus


def build_retriever(name: str, namespace: str | None = None) -> Retriever:
    """Instantiate a retriever backend without indexing it.

    Imports are local so that selecting one backend never forces the heavy
    dependencies of the others to load. ``namespace`` isolates the persistent
    storage of the Chroma-backed engines so different agents keep separate DBs.
    """
    if name == "pure-numpy":
        from localagent.rag.numpy_store import NumpyRetriever

        return NumpyRetriever()
    if name == "pure-chroma":
        from localagent.rag.chroma_store import ChromaRetriever

        return ChromaRetriever()
    if name == "llamaindex":
        from localagent.rag.llamaindex_store import LlamaIndexRetriever

        return LlamaIndexRetriever()
    if name == "llamaindex-chroma":
        from localagent.rag.llamaindex_chroma_store import LlamaIndexChromaRetriever

        return LlamaIndexChromaRetriever(namespace=namespace)
    if name == "haystack":
        from localagent.rag.haystack_store import HaystackRetriever

        return HaystackRetriever()
    if name == "haystack-chroma":
        from localagent.rag.haystack_chroma_store import HaystackChromaRetriever

        return HaystackChromaRetriever(namespace=namespace)
    if name == "rerank":
        from localagent.rag.rerank_store import RerankRetriever

        return RerankRetriever()
    raise ValueError(f"unknown retriever backend: {name}")


def build_indexed_retriever(
    name: str,
    logger: AgentLogger,
    skip_index: bool = False,
    drop: bool = False,
    namespace: str | None = None,
) -> Retriever:
    """Build a retriever, reusing or (re)building its index as requested.

    - ``drop``: clear any persisted storage first (only Chroma persists).
    - ``skip_index``: start without indexing the corpus (use whatever already exists).
    - ``namespace``: isolate persistent storage (Chroma backends) per agent.
    - Otherwise, reuse a populated persistent store, or index the corpus.
    """
    retriever = build_retriever(name, namespace=namespace)

    if drop:
        retriever.reset()
        logger.info("dropped rag storage", backend=name)

    if skip_index:
        logger.info("skipping indexing (no corpus loaded)", backend=name)
        return retriever

    if retriever.is_populated():
        logger.info("reusing persisted index", backend=name)
        return retriever

    documents = load_corpus()
    logger.info(f"indexing corpus with '{name}'", backend=name, documents=len(documents))
    retriever.index(documents)
    logger.info("corpus indexed", backend=name)
    return retriever
