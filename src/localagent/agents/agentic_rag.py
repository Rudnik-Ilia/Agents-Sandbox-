"""Entrypoint: agentic RAG - native tool-calling agent that can search the corpus.

Retrieval is exposed as a `search_knowledge_base` tool (backed by the hybrid
BM25+dense+rerank backend). The model decides per query whether to search, use
another tool (calculator, shell, web_search, MCP), or answer directly.
"""

from __future__ import annotations

from localagent.agents.args import parse_rag_flags
from localagent.agents.tools import NativeToolAgent
from localagent.cli import run_repl
from localagent.logging_setup import get_agent_logger
from localagent.rag.factory import build_indexed_retriever
from localagent.tools.retrieval import make_retrieval_tool


def main() -> None:
    """Run the agentic-RAG agent (tool-calling + retrieval-as-a-tool)."""
    args = parse_rag_flags("Agentic RAG: tool-calling agent with a knowledge-base search tool.")
    logger = get_agent_logger("agentic-rag")
    retriever = build_indexed_retriever("rerank", logger, skip_index=args.skip_index, drop=args.drop)
    agent = NativeToolAgent(logger, extra_tools=[make_retrieval_tool(retriever)], session="agentic-rag")
    agent.instructions = (
        "You are an assistant with tools: search_knowledge_base, calculator, shell, "
        "web_search (and any MCP tools). For factual questions, ALWAYS try "
        "search_knowledge_base first; only use web_search if the knowledge base lacks "
        "the answer. Use other tools when appropriate; otherwise answer directly. Be concise."
    )
    run_repl(
        agent,
        logger,
        title="Agentic RAG (tool-calling + hybrid rerank retrieval)",
        subtitle="the model decides when to search, compute, or answer directly",
    )


if __name__ == "__main__":
    main()
