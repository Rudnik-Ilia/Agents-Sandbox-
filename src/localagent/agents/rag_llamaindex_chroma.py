"""Entrypoint: RAG-only agent using LlamaIndex orchestration over a Chroma store."""

from __future__ import annotations

from localagent.agents.args import parse_rag_flags
from localagent.agents.rag import run_rag


def main() -> None:
    """Run the RAG agent with LlamaIndex orchestration on a persistent Chroma store."""
    args = parse_rag_flags("RAG agent (LlamaIndex orchestration + persistent Chroma store).")
    run_rag(backend="llamaindex-chroma", skip_index=args.skip_index, drop=args.drop)


if __name__ == "__main__":
    main()
