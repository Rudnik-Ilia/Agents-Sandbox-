"""Shared command-line flags for all agents.

Every agent supports ``--no-soul`` to skip loading SOUL.md. RAG and hybrid agents
additionally get ``--no-index`` and ``--drop``; chat agents get ``--persist``.
"""

from __future__ import annotations

import argparse

from localagent.soul import set_soul_enabled


def parse_agent_flags(description: str, *, rag: bool = False, persist: bool = False) -> argparse.Namespace:
    """Parse common agent flags and apply the SOUL toggle.

    Returns the parsed namespace. ``--no-soul`` is always available; ``rag`` adds
    ``--no-index``/``--drop``; ``persist`` adds ``--persist``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--no-soul", dest="soul", action="store_false", default=True, help="do not load SOUL.md into the system prompt")
    if persist:
        parser.add_argument("--persist", action="store_true", help="persist memory to SQLite across runs")
    if rag:
        parser.add_argument(
            "--no-index", dest="skip_index", action="store_true", help="start without indexing the corpus"
        )
        parser.add_argument(
            "--drop", action="store_true", help="drop persisted RAG storage before starting (Chroma only)"
        )

    args = parser.parse_args()
    set_soul_enabled(args.soul)
    return args


def parse_rag_flags(description: str) -> argparse.Namespace:
    """Backward-compatible helper for RAG/hybrid agents (adds soul + rag flags)."""
    return parse_agent_flags(description, rag=True)
