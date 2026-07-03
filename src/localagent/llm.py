"""Factory helpers for the local Ollama chat model and embeddings."""

from __future__ import annotations

from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama, OllamaEmbeddings

from localagent.config import get_settings


def build_chat_llm(
    temperature: float | None = None, model: str | None = None, **kwargs: object
) -> ChatOllama:
    """Create a `ChatOllama` client pointed at the configured Ollama host."""
    settings = get_settings()
    return ChatOllama(
        model=model or settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature if temperature is None else temperature,
        keep_alive=settings.llm_keep_alive,
        num_ctx=settings.llm_num_ctx,
        reasoning=settings.llm_think,
        client_kwargs={"timeout": settings.llm_request_timeout},
        **kwargs,
    )


def with_reliability(runnable: Runnable) -> Runnable:
    """Add retries and a fallback model to any chat runnable.

    Works on plain, tool-bound, or structured-output runnables: retries transient
    Ollama errors/timeouts, then falls back to the configured fallback model.
    """
    settings = get_settings()
    retried = runnable.with_retry(stop_after_attempt=max(1, settings.llm_max_retries))
    if settings.fallback_model and settings.fallback_model != settings.llm_model:
        fallback = build_chat_llm(model=settings.fallback_model).with_retry(
            stop_after_attempt=max(1, settings.llm_max_retries)
        )
        return retried.with_fallbacks([fallback])
    return retried


def with_retry_only(runnable: Runnable) -> Runnable:
    """Add retries without a fallback (use for structured-output runnables)."""
    return runnable.with_retry(stop_after_attempt=max(1, get_settings().llm_max_retries))


def build_reliable_chat(temperature: float | None = None) -> Runnable:
    """A plain chat model wrapped with retry + fallback, for generation calls."""
    return with_reliability(build_chat_llm(temperature=temperature))


def build_embeddings() -> OllamaEmbeddings:
    """Create an `OllamaEmbeddings` client for the configured embedding model."""
    settings = get_settings()
    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)


def token_usage(message: object) -> dict[str, int]:
    """Best-effort extraction of token counts from a LangChain AI message."""
    usage = getattr(message, "usage_metadata", None) or {}
    if usage:
        return {
            "input": int(usage.get("input_tokens", 0)),
            "output": int(usage.get("output_tokens", 0)),
            "total": int(usage.get("total_tokens", 0)),
        }
    meta = getattr(message, "response_metadata", None) or {}
    prompt_tokens = int(meta.get("prompt_eval_count", 0))
    output_tokens = int(meta.get("eval_count", 0))
    return {"input": prompt_tokens, "output": output_tokens, "total": prompt_tokens + output_tokens}
