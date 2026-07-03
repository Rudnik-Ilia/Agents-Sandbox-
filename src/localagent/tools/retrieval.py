"""Expose a retriever as a tool, enabling agentic RAG (the LLM decides when to search)."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from localagent.rag.base import Retriever


def make_retrieval_tool(retriever: Retriever) -> BaseTool:
    """Wrap a retriever's search as a `search_knowledge_base` tool."""

    def search_knowledge_base(query: str) -> str:
        chunks = retriever.search(query)
        if not chunks:
            return "no relevant documents found"
        return "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in chunks)

    return StructuredTool.from_function(
        func=search_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Search the local document knowledge base for facts before answering. "
            "Use this for any question that may be covered by the indexed corpus."
        ),
    )
