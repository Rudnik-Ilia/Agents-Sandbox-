"""Mini RAGAS-style evaluation harness to compare RAG backends objectively.

For a small labelled question set it measures, per backend:
- hit-rate: did retrieval return the expected source document?
- context recall: fraction of expected keywords present in the retrieved text.

Run with: ``uv run rag-eval``
"""

from __future__ import annotations

from dataclasses import dataclass

from localagent.logging_setup import get_agent_logger
from localagent.rag.factory import build_indexed_retriever

BACKENDS = ["pure-numpy", "pure-chroma", "llamaindex", "haystack", "rerank"]


@dataclass(frozen=True)
class EvalItem:
    """A labelled evaluation question."""

    question: str
    source: str
    keywords: tuple[str, ...]


DATASET: list[EvalItem] = [
    EvalItem("What is retrieval-augmented generation?", "rag_systems.md", ("retriev", "context")),
    EvalItem("Difference between buffer and summary memory?", "memory_and_tools.md", ("buffer", "summary")),
    EvalItem("How does native tool calling work?", "memory_and_tools.md", ("tool", "structured")),
    EvalItem("How to route between RAG and the LLM?", "hybrid_routing.md", ("semantic", "adaptive")),
    EvalItem("What are the building blocks of an AI agent?", "agents_overview.md", ("memory", "tools")),
]


def _evaluate(backend: str, logger) -> tuple[float, float]:
    retriever = build_indexed_retriever(backend, logger, drop=True)
    hits = 0
    recall_sum = 0.0
    for item in DATASET:
        chunks = retriever.search(item.question)
        sources = {c.source for c in chunks}
        context = " ".join(c.text for c in chunks).lower()
        if item.source in sources:
            hits += 1
        found = sum(1 for kw in item.keywords if kw.lower() in context)
        recall_sum += found / len(item.keywords)
    n = len(DATASET)
    return hits / n, recall_sum / n


def main() -> None:
    """Evaluate all configured backends and print a comparison table."""
    from localagent.approval import set_auto_approve

    set_auto_approve(True)
    logger = get_agent_logger("rag-eval")

    print(f"{'backend':<16} {'hit_rate':>10} {'ctx_recall':>12}")
    print("-" * 40)
    for backend in BACKENDS:
        try:
            hit_rate, ctx_recall = _evaluate(backend, logger)
            print(f"{backend:<16} {hit_rate:>10.2f} {ctx_recall:>12.2f}")
        except Exception as exc:  # noqa: BLE001 - report and continue with other backends
            print(f"{backend:<16} {'ERROR':>10}  {exc}")


if __name__ == "__main__":
    main()
