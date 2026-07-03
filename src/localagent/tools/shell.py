"""Tool: execute a shell command on the local machine.

Educational/local use only: this runs arbitrary commands through the system
shell. Output is captured, time-limited and truncated.
"""

from __future__ import annotations

import subprocess

from langchain_core.tools import tool

_TIMEOUT = 30
_MAX_OUTPUT = 2000


@tool
def shell(command: str) -> str:
    """Execute a shell command and return its combined stdout and stderr."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {_TIMEOUT}s"
    except OSError as exc:
        return f"error: {exc}"

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if not output:
        output = f"(exit code {proc.returncode}, no output)"
    return output[:_MAX_OUTPUT]
