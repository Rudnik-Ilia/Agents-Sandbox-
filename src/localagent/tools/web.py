"""Tool: web search via DuckDuckGo (no API key required)."""

from __future__ import annotations

from langchain_core.tools import tool

_MAX_RESULTS = 5


@tool
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo and return the top results (title, snippet, url)."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "error: the 'ddgs' package is not installed"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=_MAX_RESULTS))
    except Exception as exc:  # noqa: BLE001 - network/library errors vary widely
        return f"error: {exc}"

    if not results:
        return "no results"
    return "\n".join(
        f"- {r.get('title', '')} | {r.get('body', '')} | {r.get('href', '')}" for r in results
    )
