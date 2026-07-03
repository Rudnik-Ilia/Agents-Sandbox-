"""Entrypoint: chat agent using full conversation buffer memory."""

from __future__ import annotations

from localagent.agents.args import parse_agent_flags
from localagent.agents.chat import run_chat


def main() -> None:
    """Run the buffer-memory chat agent."""
    args = parse_agent_flags("Chat agent with full conversation buffer memory.", persist=True)
    run_chat(mode="buffer", persist=args.persist)


if __name__ == "__main__":
    main()
