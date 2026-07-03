"""Entrypoint: production-style RAG agent (hybrid BM25+dense retrieval + cross-encoder rerank)."""

from __future__ import annotations

from localagent.agents.args import parse_rag_flags
from localagent.agents.rag import run_rag


def main() -> None:
    """Run the RAG agent with hybrid retrieval and cross-encoder reranking."""
    args = parse_rag_flags("RAG agent (hybrid BM25 + dense retrieval, cross-encoder rerank).")
    run_rag(backend="rerank", skip_index=args.skip_index, drop=args.drop)


if __name__ == "__main__":
    main()
