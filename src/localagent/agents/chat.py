"""Chat agent with pluggable conversation memory.

Two entrypoints reuse this module: one selects buffer memory, the other
summary-buffer memory. Both optionally persist across runs.
"""

from __future__ import annotations

import time

from langchain_core.messages import SystemMessage

from localagent.agents.base import Agent
from localagent.cli import run_repl
from localagent.llm import build_chat_llm, build_reliable_chat, token_usage
from localagent.logging_setup import get_agent_logger
from localagent.memory import BufferMemory, ChatMemory, SummaryBufferMemory

INSTRUCTIONS = (
    "You are a friendly conversational assistant. Be concise and remember earlier "
    "details the user shared during the conversation."
)


class ChatAgent(Agent):
    """A memory-backed conversational agent."""

    instructions = INSTRUCTIONS

    def __init__(self, memory: ChatMemory, logger) -> None:
        self._llm = build_reliable_chat()
        self._memory = memory
        self._logger = logger

    def respond(self, user_text: str, skill_context: str = "") -> str:
        self._memory.add_user(user_text)
        messages = [SystemMessage(self.system_prompt(skill_context)), *self._memory.messages()]

        start = time.perf_counter()
        reply = self._llm.invoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000

        answer = reply.content.strip()
        # breakpoint()
        self._logger.llm_call(user_text, answer, latency_ms, token_usage(reply))
        self._memory.add_ai(answer)
        self._memory.save()
        return answer


def run_chat(mode: str, persist: bool) -> None:
    """Launch the chat REPL with the requested memory strategy."""
    logger = get_agent_logger(f"chat-{mode}")
    session = f"chat-{mode}"
    if mode == "summary":
        memory: ChatMemory = SummaryBufferMemory(build_chat_llm(), session=session, persist=persist)
    else:
        memory = BufferMemory(session=session, persist=persist)

    agent = ChatAgent(memory, logger)
    subtitle = f"memory={mode} persist={'on' if persist else 'off'} | type your message"
    run_repl(agent, logger, title=f"Chat agent ({mode} memory)", subtitle=subtitle)
