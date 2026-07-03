"""Entrypoint: hybrid agent (semantic router) over a Haystack + Chroma RAG backend.

Same persistence and live-ingest functionality as the RAG+Chroma agents, but the
retrieval engine is Haystack with a persistent Chroma store, and queries are
routed between RAG and plain chat by the semantic (embedding-similarity) router.
"""

from __future__ import annotations

from localagent.agents.args import parse_rag_flags
from localagent.agents.hybrid import run_hybrid


def main() -> None:
    """Run the semantic-router hybrid agent on the Haystack + Chroma backend."""
    args = parse_rag_flags("Hybrid agent (semantic router) over Haystack + persistent Chroma.")
    run_hybrid(
        router_name="semantic",
        skip_index=args.skip_index,
        drop=args.drop,
        rag_backend="haystack-chroma",
        rag_namespace="hybrid-semantic",
    )


if __name__ == "__main__":
    main()
