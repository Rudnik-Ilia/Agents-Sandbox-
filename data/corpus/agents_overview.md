# AI Agents Overview

An AI agent is a system that uses a large language model (LLM) as a reasoning
engine to decide which actions to take in order to accomplish a goal. Unlike a
plain chatbot that only produces text, an agent can call tools, retrieve
documents, keep memory of a conversation, and chain multiple steps together.

## Core building blocks

- **Model**: the LLM that generates text and decisions. In this project the model
  is served locally by Ollama.
- **Memory**: stores conversation history so the agent can stay coherent across
  turns. Common strategies are full buffer memory and summary-buffer memory.
- **Tools**: external functions the agent can invoke, such as a calculator, a web
  search, or a database query.
- **Retrieval**: the ability to fetch relevant documents from a knowledge base to
  ground the answer in facts. This is the basis of RAG.
- **Orchestration**: the control flow that ties everything together. Frameworks
  such as LangChain and LangGraph provide orchestration primitives.

## Common agent patterns

1. **Chat agent**: conversation plus memory, no external actions.
2. **Tool-calling agent**: the model decides when to call a tool and with what
   arguments. Tool use can be done with a text protocol (ReAct) or with native
   structured tool calls.
3. **RAG agent**: retrieve-then-read, where context is fetched before answering.
4. **Hybrid agent**: combines chat and RAG and decides per query whether retrieval
   is needed.

## Industry standards

Production agent stacks typically include observability (structured logging and
tracing), configuration via environment variables, evaluation of answer quality,
guardrails, and a clear separation between orchestration and model providers.
