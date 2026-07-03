"""Human-in-the-loop approval for dangerous tool calls.

When approval is required (config ``require_tool_approval``), the user must confirm
before a dangerous tool (e.g. ``shell``) runs. Auto-approve can be toggled for
non-interactive runs such as evaluation.
"""

from __future__ import annotations

from typing import Any

from localagent.config import get_settings

_auto_approve = False
_BORDER = "-" * 78


def set_auto_approve(enabled: bool) -> None:
    """Skip interactive confirmation (e.g. in tests/eval) when enabled."""
    global _auto_approve
    _auto_approve = enabled


def confirm_tool(tool_name: str, arguments: Any) -> bool:
    """Return True if the tool call is allowed to run."""
    if _auto_approve or not get_settings().require_tool_approval:
        return True
    print(_BORDER)
    print(f"  APPROVAL REQUIRED - dangerous tool '{tool_name}'")
    print(f"  arguments: {arguments}")
    print(_BORDER)
    try:
        answer = input("  run this? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}
