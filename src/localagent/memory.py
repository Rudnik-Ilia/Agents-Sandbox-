"""Conversation memory strategies with optional SQLite persistence.

Two strategies are provided for comparison:

* :class:`BufferMemory` keeps every turn verbatim.
* :class:`SummaryBufferMemory` keeps recent turns verbatim and folds older turns
  into a running LLM-generated summary to bound the context size.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from localagent.config import get_settings


def _store_path() -> Path:
    return get_settings().memory_db


def _ensure_table() -> sqlite3.Connection:
    conn = sqlite3.connect(_store_path())
    conn.execute("CREATE TABLE IF NOT EXISTS memory (session TEXT PRIMARY KEY, data TEXT NOT NULL)")
    return conn


class ChatMemory(ABC):
    """Base class for conversation memory backends."""

    def __init__(self, session: str, persist: bool) -> None:
        self.session = session
        self.persist = persist
        if persist:
            self._load()

    @abstractmethod
    def add_user(self, text: str) -> None:
        """Record a user turn."""

    @abstractmethod
    def add_ai(self, text: str) -> None:
        """Record an assistant turn."""

    @abstractmethod
    def messages(self) -> list[BaseMessage]:
        """Return the message list to feed the model as conversation context."""

    @abstractmethod
    def _state(self) -> dict:
        """Serialise internal state for persistence."""

    @abstractmethod
    def _restore(self, state: dict) -> None:
        """Restore internal state from persistence."""

    def save(self) -> None:
        """Persist current state when persistence is enabled."""
        if not self.persist:
            return
        conn = _ensure_table()
        try:
            conn.execute(
                "INSERT INTO memory (session, data) VALUES (?, ?) "
                "ON CONFLICT(session) DO UPDATE SET data = excluded.data",
                (self.session, json.dumps(self._state())),
            )
            conn.commit()
        finally:
            conn.close()

    def _load(self) -> None:
        if not _store_path().exists():
            return
        conn = _ensure_table()
        try:
            row = conn.execute("SELECT data FROM memory WHERE session = ?", (self.session,)).fetchone()
        finally:
            conn.close()
        if row:
            self._restore(json.loads(row[0]))


class BufferMemory(ChatMemory):
    """Keep the full conversation history in memory."""

    def __init__(self, session: str = "default", persist: bool = False) -> None:
        self._turns: list[tuple[str, str]] = []
        super().__init__(session, persist)

    def add_user(self, text: str) -> None:
        self._turns.append(("user", text))

    def add_ai(self, text: str) -> None:
        self._turns.append(("ai", text))

    def messages(self) -> list[BaseMessage]:
        return [HumanMessage(t) if role == "user" else AIMessage(t) for role, t in self._turns]

    def _state(self) -> dict:
        return {"turns": self._turns}

    def _restore(self, state: dict) -> None:
        self._turns = [tuple(t) for t in state.get("turns", [])]


class SummaryBufferMemory(ChatMemory):
    """Keep recent turns verbatim and summarise older turns with the LLM."""

    def __init__(
        self,
        llm: BaseChatModel,
        session: str = "default",
        persist: bool = False,
        window: int = 6,
    ) -> None:
        self._llm = llm
        self._window = window
        self._summary = ""
        self._turns: list[tuple[str, str]] = []
        super().__init__(session, persist)

    def add_user(self, text: str) -> None:
        self._turns.append(("user", text))

    def add_ai(self, text: str) -> None:
        self._turns.append(("ai", text))
        self._maybe_summarise()

    def _maybe_summarise(self) -> None:
        if len(self._turns) <= self._window:
            return
        overflow = self._turns[: -self._window]
        self._turns = self._turns[-self._window :]
        transcript = "\n".join(f"{role}: {content}" for role, content in overflow)
        prompt = (
            "Update the running summary of a conversation. Keep it concise and factual.\n\n"
            f"Existing summary:\n{self._summary or '(none)'}\n\n"
            f"New exchanges:\n{transcript}\n\nUpdated summary:"
        )
        self._summary = self._llm.invoke(prompt).content.strip()

    def messages(self) -> list[BaseMessage]:
        history: list[BaseMessage] = []
        if self._summary:
            history.append(SystemMessage(f"Summary of earlier conversation:\n{self._summary}"))
        history.extend(HumanMessage(t) if role == "user" else AIMessage(t) for role, t in self._turns)
        return history

    def _state(self) -> dict:
        return {"summary": self._summary, "turns": self._turns}

    def _restore(self, state: dict) -> None:
        self._summary = state.get("summary", "")
        self._turns = [tuple(t) for t in state.get("turns", [])]
