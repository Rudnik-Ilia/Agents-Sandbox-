"""Entrypoint: RAG-only agent using the Haystack retriever."""

from __future__ import annotations

from localagent.agents.args import parse_rag_flags
from localagent.agents.rag import run_rag


def main() -> None:
    """Run the RAG agent with the Haystack backend."""
    args = parse_rag_flags("RAG agent (Haystack retriever).")
    run_rag(backend="haystack", skip_index=args.skip_index, drop=args.drop)


if __name__ == "__main__":
    main()
