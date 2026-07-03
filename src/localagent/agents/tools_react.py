"""Entrypoint: tool-calling agent using the ReAct text protocol."""

from __future__ import annotations

from localagent.agents.args import parse_agent_flags
from localagent.agents.tools import run_tools


def main() -> None:
    """Run the ReAct-style tool agent."""
    parse_agent_flags("Tool-calling agent using the ReAct text protocol.")
    run_tools(mode="react")


if __name__ == "__main__":
    main()
