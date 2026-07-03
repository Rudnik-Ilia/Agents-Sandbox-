"""Entrypoint: adaptive-RAG hybrid agent over a LlamaIndex + Chroma backend.

Same persistence and live-ingest functionality as the RAG+Chroma agents, but the
retrieval engine is LlamaIndex with its own persistent Chroma store, and routing
is done by adaptive RAG (retrieve -> grade relevance -> answer or fall back).
"""

from __future__ import annotations

from localagent.agents.args import parse_rag_flags
from localagent.agents.hybrid import run_hybrid


def main() -> None:
    """Run the adaptive-RAG hybrid agent on the LlamaIndex + Chroma backend."""
    args = parse_rag_flags("Hybrid agent (adaptive RAG via LangGraph) over LlamaIndex + persistent Chroma.")
    run_hybrid(
        router_name="adaptive",
        skip_index=args.skip_index,
        drop=args.drop,
        rag_backend="llamaindex-chroma",
        rag_namespace="hybrid-adaptive",
    )


if __name__ == "__main__":
    main()
