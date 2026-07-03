"""Routing strategies that decide whether a query needs retrieval or plain chat."""

from __future__ import annotations

from typing import Literal

import numpy as np
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from localagent.llm import build_embeddings, with_retry_only

Route = Literal["rag", "chat"]

_RAG_EXEMPLARS = [
    "What is retrieval augmented generation?",
    "How does an AI agent use tools?",
    "Explain the difference between buffer and summary memory.",
    "What approaches exist for routing between RAG and an LLM?",
    "Describe how vector search and embeddings work.",
    "What are the components of this project?",
]
_CHAT_EXEMPLARS = [
    "Hello, how are you today?",
    "Tell me a joke.",
    "What is your name?",
    "Thanks for the help!",
    "Let's just chat for a bit.",
    "Good morning!",
]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class SemanticRouter:
    """Embed the query and route by cosine similarity to topic exemplars.

    A modern, LLM-free router: it compares the query against pre-embedded
    "retrieval" and "chit-chat" exemplar centroids and picks the closer one.
    """

    def __init__(self) -> None:
        embeddings = build_embeddings()
        rag_vecs = _normalize(np.array(embeddings.embed_documents(_RAG_EXEMPLARS), dtype=np.float32))
        chat_vecs = _normalize(np.array(embeddings.embed_documents(_CHAT_EXEMPLARS), dtype=np.float32))
        self._rag_centroid = _normalize(rag_vecs.mean(axis=0, keepdims=True))[0]
        self._chat_centroid = _normalize(chat_vecs.mean(axis=0, keepdims=True))[0]
        self._embeddings = embeddings

    def route(self, query: str) -> tuple[Route, str]:
        """Return the chosen route and a short explanation."""
        qv = np.array(self._embeddings.embed_query(query), dtype=np.float32)
        qv /= max(float(np.linalg.norm(qv)), 1e-12)
        sim_rag = float(qv @ self._rag_centroid)
        sim_chat = float(qv @ self._chat_centroid)
        detail = f"sim_rag={sim_rag:.3f} sim_chat={sim_chat:.3f}"
        return ("rag" if sim_rag >= sim_chat else "chat"), detail


class RouteDecision(BaseModel):
    """Structured output schema for the LLM-based router."""

    route: Route = Field(description="'rag' if the question needs document lookup, else 'chat'")
    reason: str = Field(description="brief justification")


_LLM_ROUTER_PROMPT = (
    "Decide how to handle the user's message. Choose 'rag' when answering needs "
    "factual knowledge that should be looked up in a document corpus about AI agents, "
    "RAG, memory and tools. Choose 'chat' for greetings, small talk, or general "
    "conversation.\n\nUser message: {query}"
)


class LLMRouter:
    """Ask the LLM to classify the query using structured (JSON) output."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._classifier = with_retry_only(llm.with_structured_output(RouteDecision))

    def route(self, query: str) -> tuple[Route, str]:
        decision: RouteDecision = self._classifier.invoke(_LLM_ROUTER_PROMPT.format(query=query))
        return decision.route, decision.reason
