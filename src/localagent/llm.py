"""Factory helpers for chat models and local Ollama embeddings."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

from localagent.config import get_settings


def _openrouter_headers() -> dict[str, str]:
    settings = get_settings()
    headers = {}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_name:
        headers["X-Title"] = settings.openrouter_app_name
    return headers


def build_chat_llm(
    temperature: float | None = None, model: str | None = None, **kwargs: object
) -> BaseChatModel:
    """Create a chat model client for the configured provider."""
    settings = get_settings()
    provider = settings.llm_provider.lower().strip()
    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        return ChatOpenAI(
            model=model or settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=settings.llm_temperature if temperature is None else temperature,
            timeout=settings.llm_request_timeout,
            default_headers=_openrouter_headers() or None,
        )
    if provider != "ollama":
        raise ValueError(f"unknown LLM_PROVIDER: {settings.llm_provider}")
    return ChatOllama(
        model=model or settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature if temperature is None else temperature,
        keep_alive=settings.llm_keep_alive,
        num_ctx=settings.llm_num_ctx,
        reasoning=settings.llm_think,
        client_kwargs={"timeout": settings.llm_request_timeout},
    )


def with_reliability(runnable: Runnable) -> Runnable:
    """Add retries and a fallback model to any chat runnable.

    Works on plain, tool-bound, or structured-output runnables: retries transient
    errors/timeouts, then falls back to the configured fallback model.
    """
    settings = get_settings()
    retried = runnable.with_retry(stop_after_attempt=max(1, settings.llm_max_retries))
    provider = settings.llm_provider.lower().strip()
    primary_model = settings.openrouter_model if provider == "openrouter" else settings.llm_model
    if settings.fallback_model and settings.fallback_model != primary_model:
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
