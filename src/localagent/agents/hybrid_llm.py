"""Entrypoint: hybrid agent (LLM classifier router) over a LlamaIndex + Chroma backend.

Same persistence and live-ingest functionality as the RAG+Chroma agents, but the
retrieval engine is LlamaIndex with a persistent Chroma store, and queries are
routed between RAG and plain chat by the LLM classifier router.
"""

from __future__ import annotations

from localagent.agents.args import parse_rag_flags
from localagent.agents.hybrid import run_hybrid


def main() -> None:
    """Run the LLM-router hybrid agent on the LlamaIndex + Chroma backend."""
    args = parse_rag_flags("Hybrid agent (LLM classifier router) over LlamaIndex + persistent Chroma.")
    run_hybrid(
        router_name="llm",
        skip_index=args.skip_index,
        drop=args.drop,
        rag_backend="llamaindex-chroma",
        rag_namespace="hybrid-llm",
    )


if __name__ == "__main__":
    main()
