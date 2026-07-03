"""Tool: let the agent persist a fact to long-term memory (SOUL.md)."""

from __future__ import annotations

from langchain_core.tools import tool

from localagent.soul import append_memory


@tool
def remember(note: str) -> str:
    """Save an important, durable fact to long-term memory for future sessions."""
    return append_memory(note)
