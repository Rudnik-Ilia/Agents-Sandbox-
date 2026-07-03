"""Entrypoint: corrective-RAG hybrid agent (multi-grader + query rewrite).

A Self-RAG / CRAG style flow over LlamaIndex + its own persistent Chroma store:
retrieve -> grade each document -> generate -> grade hallucination + answer ->
rewrite the query and retry (bounded) or fall back to plain chat.
"""

from __future__ import annotations

from localagent.agents.args import parse_rag_flags
from localagent.agents.hybrid import run_hybrid


def main() -> None:
    """Run the corrective-RAG hybrid agent on the LlamaIndex + Chroma backend."""
    args = parse_rag_flags("Hybrid agent (corrective RAG: multi-grader + query rewrite).")
    run_hybrid(
        router_name="adaptive-plus",
        skip_index=args.skip_index,
        drop=args.drop,
        rag_backend="llamaindex-chroma",
        rag_namespace="hybrid-adaptive-plus",
    )


if __name__ == "__main__":
    main()
