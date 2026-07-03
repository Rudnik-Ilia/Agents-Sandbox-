"""Global persona file (SOUL.md) that can be loaded into every agent's prompt.

A process-wide toggle (set from the CLI ``--no-soul`` flag) controls whether the
file is injected. When enabled and present, its text is prepended to the system
prompt of every agent.
"""

from __future__ import annotations

from localagent.config import PROJECT_ROOT

_enabled = True

_MEMORY_START = "<!-- memory:start -->"
_MEMORY_END = "<!-- memory:end -->"
_MAX_MEMORY = 50


def _soul_path():
    return PROJECT_ROOT / "SOUL.md"


def set_soul_enabled(enabled: bool) -> None:
    """Enable or disable loading of SOUL.md for this process."""
    global _enabled
    _enabled = enabled


def soul_text() -> str:
    """Return SOUL.md contents when enabled and present, else an empty string."""
    if not _enabled:
        return ""
    path = _soul_path()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def append_memory(note: str) -> str:
    """Append a fact to the managed Memory section of SOUL.md (append-only, deduped).

    Writes only inside the ``memory:start``/``memory:end`` markers so the curated
    persona is never overwritten. Keeps at most the most recent entries.
    """
    note = " ".join(note.split())
    if not note:
        return "nothing to remember"

    path = _soul_path()
    text = path.read_text(encoding="utf-8") if path.exists() else "# Soul\n"
    if _MEMORY_END not in text:
        text = f"{text.rstrip()}\n\n## Memory (learned)\n{_MEMORY_START}\n{_MEMORY_END}\n"

    pre, _, rest = text.partition(_MEMORY_START)
    block, _, post = rest.partition(_MEMORY_END)
    items = [line for line in block.strip().splitlines() if line.strip().startswith("- ")]

    entry = f"- {note}"
    if entry in items:
        return "already remembered"
    items.append(entry)
    items = items[-_MAX_MEMORY:]

    new_text = f"{pre}{_MEMORY_START}\n" + "\n".join(items) + f"\n{_MEMORY_END}{post}"
    path.write_text(new_text, encoding="utf-8")
    return f"remembered: {note}"
