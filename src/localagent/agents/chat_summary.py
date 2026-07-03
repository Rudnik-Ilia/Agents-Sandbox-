"""Entrypoint: chat agent using summary-buffer memory."""

from __future__ import annotations

from localagent.agents.args import parse_agent_flags
from localagent.agents.chat import run_chat


def main() -> None:
    """Run the summary-buffer chat agent."""
    args = parse_agent_flags("Chat agent with summary-buffer memory.", persist=True)
    run_chat(mode="summary", persist=args.persist)


if __name__ == "__main__":
    main()
