"""Entrypoint: tool-calling agent using native LangChain bind_tools."""

from __future__ import annotations

from localagent.agents.args import parse_agent_flags
from localagent.agents.tools import run_tools


def main() -> None:
    """Run the native tool-calling agent."""
    parse_agent_flags("Tool-calling agent using native bind_tools.")
    run_tools(mode="native")


if __name__ == "__main__":
    main()
