"""Common retriever contract plus corpus loading and chunking helpers.

Every RAG backend (pure numpy, Chroma, LlamaIndex, Haystack) implements
:class:`Retriever` so the agents can swap them without code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from localagent.config import get_settings


@dataclass(frozen=True)
class Document:
    """A raw source document before chunking."""

    source: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from a retrieval query with its relevance score."""

    text: str
    source: str
    score: float

    def as_log(self) -> dict[str, object]:
        """Compact representation for structured logs."""
        preview = self.text.replace("\n", " ")[:120]
        return {"source": self.source, "score": round(self.score, 4), "preview": preview}


class Retriever(ABC):
    """Index a corpus and return the most relevant chunks for a query."""

    name: str = "retriever"

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """Ingest and index the given documents."""

    @abstractmethod
    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks most relevant to ``query``."""

    def is_populated(self) -> bool:
        """Whether a persisted index already holds data (only true for Chroma)."""
        return False

    def reset(self) -> None:  # noqa: B027 - optional hook, only Chroma overrides
        """Drop any persisted storage. No-op for in-memory backends."""

    def add_document(self, document: Document) -> int:
        """Add a single document to the live index and return the chunk count.

        Backends that do not support live additions raise ``NotImplementedError``.
        """
        raise NotImplementedError(f"{self.name} does not support live document add")

    def known_sources(self) -> set[str]:
        """Return the set of source filenames already indexed (for corpus sync)."""
        return set()


def load_corpus(directory: Path | None = None) -> list[Document]:
    """Load every ``.md`` and ``.txt`` file in the corpus directory."""
    directory = directory or get_settings().corpus_dir
    documents: list[Document] = []
    if not directory.exists():
        return documents
    for path in sorted([*directory.glob("*.md"), *directory.glob("*.txt")]):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(source=path.name, text=text))
    return documents


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping character windows.

    A deliberately simple, dependency-free chunker used by the "pure" backends.
    """
    settings = get_settings()
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    if size <= 0:
        raise ValueError("size must be positive")
    step = max(size - overlap, 1)

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
