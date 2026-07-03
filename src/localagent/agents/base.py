"""Shared agent contract and system-prompt composition."""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage, SystemMessage

from localagent.rules import load_rules
from localagent.soul import append_memory, soul_text


class Agent(ABC):
    """Minimal contract every runnable agent implements."""

    #: Short instruction block describing this agent's behaviour.
    instructions: str = "You are a helpful assistant."

    def system_prompt(self, skill_context: str = "") -> str:
        """Compose the system prompt from always-on rules, skills and instructions."""
        return compose_system_prompt(self.instructions, skill_context)

    def context_window(self, skill_context: str = "") -> list[BaseMessage]:
        """Return the messages that would currently be sent to the model.

        Stateless agents return just the system prompt; agents with memory
        override this to include the conversation history.
        """
        return [SystemMessage(self.system_prompt(skill_context))]

    @abstractmethod
    def respond(self, user_text: str, skill_context: str = "") -> str:
        """Produce a reply to a single user message."""

    def ingest_document(self, path: str) -> str:
        """Add a document to the agent's retriever at runtime (RAG agents only)."""
        return "this agent has no document store to add to"

    def remember(self, note: str) -> str:
        """Persist a fact to the SOUL.md memory section (available to every agent)."""
        return append_memory(note)


def compose_system_prompt(instructions: str, skill_context: str = "") -> str:
    """Build a system prompt: project rules first, then loaded skills, then instructions."""
    parts: list[str] = []
    soul = soul_text()
    if soul:
        parts.append(soul)
    rules = load_rules()
    if rules:
        parts.append(f"# Project rules (always apply)\n{rules}")
    if skill_context.strip():
        parts.append(f"# Active skill instructions\n{skill_context.strip()}")
    parts.append(f"# Role\n{instructions.strip()}")
    return "\n\n".join(parts)
