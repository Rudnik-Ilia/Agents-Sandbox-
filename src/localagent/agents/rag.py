"""RAG-only agent: every query is answered strictly from retrieved context."""

from __future__ import annotations

import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from localagent.agents.base import Agent
from localagent.cli import run_repl
from localagent.llm import build_reliable_chat, token_usage
from localagent.logging_setup import get_agent_logger
from localagent.rag.base import Document, RetrievedChunk, Retriever, load_corpus
from localagent.rag.factory import build_indexed_retriever


def _add_file(retriever: Retriever, logger, file_path: Path) -> str:
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        return f"error: {exc}"
    if not text:
        return "error: file is empty"
    try:
        added = retriever.add_document(Document(source=file_path.name, text=text))
    except NotImplementedError as exc:
        return str(exc)
    logger.info("document added", source=file_path.name, chunks=added)
    return f"added {added} chunks from {file_path.name}"


def _sync_corpus(retriever: Retriever, logger) -> str:
    """Add any documents in data/corpus that are not already indexed."""
    known = retriever.known_sources()
    new_docs = [doc for doc in load_corpus() if doc.source not in known]
    if not new_docs:
        return "no new documents found in data/corpus"
    total = 0
    names: list[str] = []
    for doc in new_docs:
        try:
            added = retriever.add_document(doc)
        except NotImplementedError as exc:
            return str(exc)
        total += added
        names.append(f"{doc.source}({added})")
    logger.info("corpus synced", new_documents=len(new_docs), chunks=total)
    return f"added {total} chunks from {len(new_docs)} new doc(s): {', '.join(names)}"


def ingest_into(retriever: Retriever, logger, path: str) -> str:
    """Add a document to a live retriever.

    With a valid file path, that file is added. With no path (or a path that does
    not point to a readable file), any new files in data/corpus are ingested.
    """
    cleaned = path.strip().strip("\"'")
    if cleaned:
        file_path = Path(cleaned).expanduser()
        if file_path.is_file():
            return _add_file(retriever, logger, file_path)
        logger.info("path not usable; syncing data/corpus instead", path=cleaned)
    return _sync_corpus(retriever, logger)

INSTRUCTIONS = (
    "You answer questions using only the provided context. If the context does not "
    "contain the answer, say you don't know. Cite sources in brackets like [source]."
)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)


class RagAgent(Agent):
    """Retrieve-then-read agent backed by a swappable retriever."""

    instructions = INSTRUCTIONS

    def __init__(self, retriever: Retriever, logger) -> None:
        self._llm = build_reliable_chat()
        self._retriever = retriever
        self._logger = logger

    def ingest_document(self, path: str) -> str:
        return ingest_into(self._retriever, self._logger, path)

    def respond(self, user_text: str, skill_context: str = "") -> str:
        chunks = self._retriever.search(user_text)
        self._logger.retrieval(user_text, [c.as_log() for c in chunks])
        if not chunks:
            return "I don't know - no relevant context was found."

        prompt = f"Context:\n{_format_context(chunks)}\n\nQuestion: {user_text}"
        messages = [SystemMessage(self.system_prompt(skill_context)), HumanMessage(prompt)]

        start = time.perf_counter()
        reply = self._llm.invoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000
        answer = reply.content.strip()
        self._logger.llm_call(prompt, answer, latency_ms, token_usage(reply))
        return answer


def run_rag(backend: str, skip_index: bool = False, drop: bool = False) -> None:
    """Launch the RAG-only REPL using the given retriever backend."""
    logger = get_agent_logger(f"rag-{backend}")
    retriever = build_indexed_retriever(backend, logger, skip_index=skip_index, drop=drop)
    agent = RagAgent(retriever, logger)
    run_repl(agent, logger, title=f"RAG agent ({backend})", subtitle="ask about the indexed corpus")
