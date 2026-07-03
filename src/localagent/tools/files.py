"""Tool: read a local text file."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

_MAX_CHARS = 4000


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file and return up to ~4000 characters of its content."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return f"error: file not found: {path}"
    if not file_path.is_file():
        return f"error: not a file: {path}"
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"error: {exc}"

    if len(text) > _MAX_CHARS:
        return text[:_MAX_CHARS] + "\n...[truncated]"
    return text
