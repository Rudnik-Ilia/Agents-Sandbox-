"""Hybrid agents that combine chat and RAG, differing only in the routing mechanism.

Three strategies are provided so they can be compared:

* ``semantic``  - embedding-similarity router (no extra LLM call).
* ``llm``       - LLM classifier with structured output.
* ``adaptive``  - LangGraph flow that retrieves, grades relevance, then answers
                  from documents or falls back to plain chat.
"""

from __future__ import annotations

import time
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from localagent.agents.base import Agent, compose_system_prompt
from localagent.agents.rag import ingest_into
from localagent.agents.router import LLMRouter, SemanticRouter
from localagent.cli import run_repl
from localagent.llm import build_chat_llm, build_reliable_chat, token_usage, with_retry_only
from localagent.logging_setup import AgentLogger, get_agent_logger
from localagent.memory import BufferMemory
from localagent.rag.base import RetrievedChunk, Retriever
from localagent.rag.factory import build_indexed_retriever

_CHAT_INSTRUCTIONS = "You are a friendly conversational assistant. Be concise."
_RAG_INSTRUCTIONS = (
    "Answer the question using only the provided context. If the context is "
    "insufficient, say so. Cite sources in brackets like [source]."
)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)


class _HybridBase(Agent):
    """Shared chat/RAG answer generation and conversation memory."""

    def __init__(self, retriever: Retriever, logger: AgentLogger) -> None:
        self._llm = build_reliable_chat()
        self._retriever = retriever
        self._logger = logger
        self._memory = BufferMemory(session="hybrid", persist=False)

    def _invoke(self, messages: list[BaseMessage], prompt_label: str) -> str:
        start = time.perf_counter()
        reply = self._llm.invoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000
        answer = reply.content.strip()
        self._logger.llm_call(prompt_label, answer, latency_ms, token_usage(reply))
        return answer

    def _chat_answer(self, user_text: str, history: list[BaseMessage], skill_context: str) -> str:
        messages = [
            SystemMessage(compose_system_prompt(_CHAT_INSTRUCTIONS, skill_context)),
            *history,
            HumanMessage(user_text),
        ]
        return self._invoke(messages, user_text)

    def _rag_answer(
        self, user_text: str, history: list[BaseMessage], skill_context: str, chunks: list[RetrievedChunk]
    ) -> str:
        prompt = f"Context:\n{_format_context(chunks)}\n\nQuestion: {user_text}"
        messages = [
            SystemMessage(compose_system_prompt(_RAG_INSTRUCTIONS, skill_context)),
            *history,
            HumanMessage(prompt),
        ]
        return self._invoke(messages, prompt)

    def _finish(self, user_text: str, answer: str) -> str:
        self._memory.add_user(user_text)
        self._memory.add_ai(answer)
        return answer

    def context_window(self, skill_context: str = "") -> list[BaseMessage]:
        return [SystemMessage(compose_system_prompt(_CHAT_INSTRUCTIONS, skill_context)), *self._memory.messages()]

    def ingest_document(self, path: str) -> str:
        return ingest_into(self._retriever, self._logger, path)


class RoutedHybridAgent(_HybridBase):
    """Hybrid agent driven by an explicit up-front router (semantic or LLM)."""

    instructions = _CHAT_INSTRUCTIONS

    def __init__(self, retriever: Retriever, logger: AgentLogger, router: SemanticRouter | LLMRouter) -> None:
        super().__init__(retriever, logger)
        self._router = router

    def respond(self, user_text: str, skill_context: str = "") -> str:
        history = self._memory.messages()
        route, detail = self._router.route(user_text)
        self._logger.route(route, detail)

        if route == "rag":
            chunks = self._retriever.search(user_text)
            self._logger.retrieval(user_text, [c.as_log() for c in chunks])
            answer = self._rag_answer(user_text, history, skill_context, chunks)
        else:
            answer = self._chat_answer(user_text, history, skill_context)
        return self._finish(user_text, answer)


class _Grade(BaseModel):
    """Whether retrieved documents are relevant to the question."""

    relevant: bool = Field(description="true if any document helps answer the question, even partially")
    reason: str = Field(description="brief justification")


class _AdaptiveState(TypedDict):
    """State threaded through the adaptive LangGraph flow."""

    query: str
    skill_context: str
    history: list[BaseMessage]
    chunks: list[RetrievedChunk]
    answer: str


class AdaptiveHybridAgent(_HybridBase):
    """Adaptive RAG: retrieve, grade relevance, then answer or fall back to chat."""

    instructions = _CHAT_INSTRUCTIONS

    def __init__(self, retriever: Retriever, logger: AgentLogger) -> None:
        super().__init__(retriever, logger)
        self._grader = with_retry_only(build_chat_llm(temperature=0).with_structured_output(_Grade))
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_AdaptiveState)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("generate", self._node_generate)
        graph.add_node("fallback", self._node_fallback)
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._node_grade,
            {"relevant": "generate", "irrelevant": "fallback"},
        )
        graph.add_edge("generate", END)
        graph.add_edge("fallback", END)
        return graph.compile()

    def _node_retrieve(self, state: _AdaptiveState) -> dict:
        chunks = self._retriever.search(state["query"])
        self._logger.retrieval(state["query"], [c.as_log() for c in chunks])
        return {"chunks": chunks}

    def _node_grade(self, state: _AdaptiveState) -> str:
        chunks = state["chunks"]
        if not chunks:
            self._logger.route("chat", "no documents retrieved")
            return "irrelevant"
        grade: _Grade = self._grader.invoke(
            "You grade whether retrieved documents can help answer a question.\n\n"
            f"Question: {state['query']}\n\nDocuments:\n{_format_context(chunks)}\n\n"
            "Set relevant=true if ANY document contains information that helps answer the "
            "question, even partially, indirectly, or by mentioning its subject. Set "
            "relevant=false only if the documents are clearly unrelated to the question."
        )
        decision = "relevant" if grade.relevant else "irrelevant"
        self._logger.route("rag" if grade.relevant else "chat", f"grade={decision}: {grade.reason}")
        return decision

    def _node_generate(self, state: _AdaptiveState) -> dict:
        answer = self._rag_answer(state["query"], state["history"], state["skill_context"], state["chunks"])
        return {"answer": answer}

    def _node_fallback(self, state: _AdaptiveState) -> dict:
        answer = self._chat_answer(state["query"], state["history"], state["skill_context"])
        return {"answer": answer}

    def respond(self, user_text: str, skill_context: str = "") -> str:
        state: _AdaptiveState = {
            "query": user_text,
            "skill_context": skill_context,
            "history": self._memory.messages(),
            "chunks": [],
            "answer": "",
        }
        result = self._graph.invoke(state)
        return self._finish(user_text, result["answer"])


class _DocGrade(BaseModel):
    """Per-document relevance decision."""

    relevant: bool = Field(description="true if this document helps answer the question, even partially")


class _Hallucination(BaseModel):
    """Whether an answer is grounded in the supporting documents."""

    grounded: bool = Field(description="true if the answer is supported by the documents")
    reason: str = Field(description="brief justification")


class _AnswerUseful(BaseModel):
    """Whether an answer actually addresses the question."""

    answers: bool = Field(description="true if the answer directly responds to the question (not a refusal)")
    reason: str = Field(description="brief justification")


_MAX_RETRIES = 2


class _AdaptivePlusState(TypedDict):
    """State threaded through the corrective (Self-RAG style) flow."""

    query: str
    original_query: str
    skill_context: str
    history: list[BaseMessage]
    chunks: list[RetrievedChunk]
    answer: str
    attempts: int
    grounded: bool
    answers: bool


class AdaptivePlusHybridAgent(_HybridBase):
    """Corrective RAG: per-doc grading, generation, hallucination + answer grading, query rewrite."""

    instructions = _CHAT_INSTRUCTIONS

    def __init__(self, retriever: Retriever, logger: AgentLogger) -> None:
        super().__init__(retriever, logger)
        grader_llm = build_chat_llm(temperature=0)
        self._doc_grader = with_retry_only(grader_llm.with_structured_output(_DocGrade))
        self._hallucination_grader = with_retry_only(grader_llm.with_structured_output(_Hallucination))
        self._answer_grader = with_retry_only(grader_llm.with_structured_output(_AnswerUseful))
        self._rewriter = build_reliable_chat(temperature=0)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_AdaptivePlusState)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("grade_documents", self._node_grade_documents)
        graph.add_node("generate", self._node_generate)
        graph.add_node("grade_generation", self._node_grade_generation)
        graph.add_node("transform_query", self._node_transform_query)
        graph.add_node("fallback", self._node_fallback)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            self._decide_after_documents,
            {"generate": "generate", "rewrite": "transform_query", "fallback": "fallback"},
        )
        graph.add_edge("generate", "grade_generation")
        graph.add_conditional_edges(
            "grade_generation",
            self._decide_after_generation,
            {"useful": END, "rewrite": "transform_query", "fallback": "fallback"},
        )
        graph.add_edge("transform_query", "retrieve")
        graph.add_edge("fallback", END)
        return graph.compile()

    def _node_retrieve(self, state: _AdaptivePlusState) -> dict:
        chunks = self._retriever.search(state["query"])
        self._logger.retrieval(state["query"], [c.as_log() for c in chunks])
        return {"chunks": chunks}

    def _node_grade_documents(self, state: _AdaptivePlusState) -> dict:
        relevant: list[RetrievedChunk] = []
        for chunk in state["chunks"]:
            grade: _DocGrade = self._doc_grader.invoke(
                "Grade if the document helps answer the question. relevant=true if it helps "
                "even partially or mentions the subject.\n\n"
                f"Question: {state['original_query']}\n\nDocument [{chunk.source}]:\n{chunk.text}"
            )
            if grade.relevant:
                relevant.append(chunk)
        self._logger.route("grade_docs", f"{len(relevant)}/{len(state['chunks'])} chunks relevant")
        return {"chunks": relevant}

    def _decide_after_documents(self, state: _AdaptivePlusState) -> str:
        if state["chunks"]:
            return "generate"
        if state["attempts"] < _MAX_RETRIES:
            return "rewrite"
        self._logger.route("chat", "no relevant docs after retries")
        return "fallback"

    def _node_generate(self, state: _AdaptivePlusState) -> dict:
        answer = self._rag_answer(
            state["original_query"], state["history"], state["skill_context"], state["chunks"]
        )
        return {"answer": answer}

    def _node_grade_generation(self, state: _AdaptivePlusState) -> dict:
        context = _format_context(state["chunks"])
        hallucination: _Hallucination = self._hallucination_grader.invoke(
            "Is the answer grounded in and supported by the documents?\n\n"
            f"Documents:\n{context}\n\nAnswer:\n{state['answer']}"
        )
        useful: _AnswerUseful = self._answer_grader.invoke(
            "Judge ONLY whether the answer directly responds to the question. "
            "Set answers=true if it gives a direct, on-topic response. Set answers=false "
            "ONLY if it refuses, says it does not know, asks for clarification, or is "
            "off-topic. Do NOT judge whether the answer is correct or how it was obtained.\n\n"
            f"Question: {state['original_query']}\n\nAnswer:\n{state['answer']}"
        )
        self._logger.route(
            "grade_generation",
            f"grounded={hallucination.grounded} answers={useful.answers}: {useful.reason}",
        )
        return {"grounded": hallucination.grounded, "answers": useful.answers}

    def _decide_after_generation(self, state: _AdaptivePlusState) -> str:
        if state["grounded"] and state["answers"]:
            return "useful"
        if state["attempts"] < _MAX_RETRIES:
            return "rewrite"
        self._logger.route("chat", "answer not grounded/useful after retries")
        return "fallback"

    def _node_transform_query(self, state: _AdaptivePlusState) -> dict:
        rewritten = self._rewriter.invoke(
            "Rewrite the question into a better search query for retrieving relevant documents. "
            "Return only the rewritten question.\n\n"
            f"Question: {state['original_query']}"
        ).content.strip()
        self._logger.route("rewrite", f"attempt {state['attempts'] + 1}: {rewritten}")
        return {"query": rewritten, "attempts": state["attempts"] + 1}

    def _node_fallback(self, state: _AdaptivePlusState) -> dict:
        answer = self._chat_answer(state["original_query"], state["history"], state["skill_context"])
        return {"answer": answer}

    def respond(self, user_text: str, skill_context: str = "") -> str:
        state: _AdaptivePlusState = {
            "query": user_text,
            "original_query": user_text,
            "skill_context": skill_context,
            "history": self._memory.messages(),
            "chunks": [],
            "answer": "",
            "attempts": 0,
            "grounded": False,
            "answers": False,
        }
        result = self._graph.invoke(state)
        return self._finish(user_text, result["answer"])


def run_hybrid(
    router_name: str,
    skip_index: bool = False,
    drop: bool = False,
    rag_backend: str = "pure-numpy",
    rag_namespace: str | None = None,
) -> None:
    """Launch the hybrid REPL with the chosen routing strategy and RAG backend."""
    logger = get_agent_logger(f"hybrid-{router_name}")
    retriever = build_indexed_retriever(
        rag_backend, logger, skip_index=skip_index, drop=drop, namespace=rag_namespace
    )

    if router_name == "semantic":
        agent: Agent = RoutedHybridAgent(retriever, logger, SemanticRouter())
        title = "Hybrid agent (semantic router)"
    elif router_name == "llm":
        agent = RoutedHybridAgent(retriever, logger, LLMRouter(build_chat_llm(temperature=0)))
        title = "Hybrid agent (LLM classifier router)"
    elif router_name == "adaptive":
        agent = AdaptiveHybridAgent(retriever, logger)
        title = "Hybrid agent (adaptive RAG via LangGraph)"
    elif router_name == "adaptive-plus":
        agent = AdaptivePlusHybridAgent(retriever, logger)
        title = "Hybrid agent (corrective RAG: multi-grader + query rewrite)"
    else:
        raise ValueError(f"unknown router: {router_name}")

    run_repl(agent, logger, title=title, subtitle="mixes small talk and corpus questions")
