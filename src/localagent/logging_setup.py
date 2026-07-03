"""Dual-channel logging: readable console output plus structured JSON log file.

Use :func:`get_agent_logger` to obtain an :class:`AgentLogger` that records the
events worth monitoring: LLM calls, tool calls, retrieval hits, routing
decisions and errors.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Any

from localagent.config import get_settings


def init_observability() -> None:
    """Enable LangSmith tracing from settings (no-op unless configured)."""
    settings = get_settings()
    if not settings.langsmith_tracing:
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)

_CONFIGURED: set[str] = set()
_BORDER = "-" * 78


def configure_console() -> None:
    """Force UTF-8 console output so model replies with non-ASCII never crash."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


class _JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON line, including structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "agent": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if isinstance(event, dict):
            payload.update(event)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _ConsoleFormatter(logging.Formatter):
    """Compact human-readable console formatter without colors or emojis."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.now().strftime("%H:%M:%S")
        kind = getattr(record, "kind", record.levelname.lower())
        return f"[{stamp}] {record.name} | {kind:<9} | {record.getMessage()}"


def _build_logger(agent: str) -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger(agent)
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False

    if agent in _CONFIGURED:
        return logger

    console = logging.StreamHandler()
    console.setFormatter(_ConsoleFormatter())
    logger.addHandler(console)

    settings.log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.log_path / f"{agent}.jsonl",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    _CONFIGURED.add(agent)
    return logger


def _truncate(text: str, limit: int = 600) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else f"{text[:limit]}..."


class AgentLogger:
    """Thin wrapper that emits both readable and structured records for key events."""

    def __init__(self, agent: str) -> None:
        self.agent = agent
        self._log = _build_logger(agent)

    def banner(self, title: str, subtitle: str = "") -> None:
        """Print a bordered header to the console (file gets a structured note)."""
        print(_BORDER)
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print(_BORDER)
        self._log.info("session start", extra={"kind": "session", "event": {"event_type": "session", "title": title}})

    def info(self, message: str, **fields: Any) -> None:
        self._log.info(message, extra={"kind": "info", "event": {"event_type": "info", **fields}})

    def llm_call(self, prompt: str, response: str, latency_ms: float, tokens: dict[str, int] | None = None) -> None:
        tok = tokens or {}
        num_ctx = get_settings().llm_num_ctx
        used = tok.get("input", 0)
        pct = (used / num_ctx * 100) if num_ctx else 0
        self._log.info(
            f"llm {latency_ms:.0f}ms | tokens in={tok.get('input', '?')} out={tok.get('output', '?')} "
            f"total={tok.get('total', '?')} | ctx {used}/{num_ctx} ({pct:.0f}%)",
            extra={
                "kind": "llm",
                "event": {
                    "event_type": "llm_call",
                    "prompt": _truncate(prompt),
                    "response": _truncate(response),
                    "latency_ms": round(latency_ms, 1),
                    "tokens": tok,
                    "context_window": num_ctx,
                    "context_used_pct": round(pct, 1),
                },
            },
        )

    def tool_call(self, name: str, arguments: Any, result: Any) -> None:
        self._log.info(
            f"tool '{name}' args={arguments} -> {result}",
            extra={
                "kind": "tool",
                "event": {"event_type": "tool_call", "tool": name, "arguments": arguments, "result": result},
            },
        )

    def retrieval(self, query: str, hits: list[dict[str, Any]]) -> None:
        preview = ", ".join(f"{h.get('source', '?')}:{h.get('score', 0):.3f}" for h in hits)
        self._log.info(
            f"retrieved {len(hits)} chunks [{preview}]",
            extra={
                "kind": "retrieval",
                "event": {"event_type": "retrieval", "query": _truncate(query, 200), "hits": hits},
            },
        )

    def route(self, decision: str, detail: str = "") -> None:
        self._log.info(
            f"route -> {decision} {('(' + detail + ')') if detail else ''}".strip(),
            extra={"kind": "route", "event": {"event_type": "route", "decision": decision, "detail": detail}},
        )

    def error(self, message: str, exc: BaseException | None = None) -> None:
        self._log.error(message, exc_info=exc, extra={"kind": "error", "event": {"event_type": "error"}})


def get_agent_logger(agent: str) -> AgentLogger:
    """Return an :class:`AgentLogger` bound to the given agent name."""
    configure_console()
    init_observability()
    return AgentLogger(agent)
