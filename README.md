# LocalAgent

An educational lab of AI agent architectures, built on **LangChain** and a local
**Ollama** server. Each variant is a separate, runnable CLI so you can study one
mechanism at a time and watch it work through the logs.

![title](/png.jpg)

- Model (generation): `gemma4:e4b`
- Model (embeddings): `mxbai-embed-large:latest`
- Host: `http://10.100.102.10:11434` (configurable)

## What is included

| Category | Variant | Command |
| --- | --- | --- |
| Chat + memory | full buffer | `chat-buffer` |
| Chat + memory | summary buffer | `chat-summary` |
| Tool calling | ReAct text protocol | `tools-react` |
| Tool calling | native `bind_tools` | `tools-native` |
| RAG only | pure numpy cosine | `rag-numpy` |
| RAG only | Chroma vector DB | `rag-chroma` |
| RAG only | LlamaIndex (in-memory) | `rag-llamaindex` |
| RAG only | LlamaIndex + Chroma store | `rag-llamaindex-chroma` |
| RAG only | Haystack | `rag-haystack` |
| RAG only | hybrid BM25+dense + cross-encoder rerank | `rag-rerank` |
| Hybrid (chat + RAG) | semantic router | `hybrid-semantic` |
| Hybrid (chat + RAG) | LLM classifier router | `hybrid-llm` |
| Hybrid (chat + RAG) | adaptive RAG (LangGraph) | `hybrid-adaptive` |
| Hybrid (chat + RAG) | corrective RAG (multi-grader + rewrite) | `hybrid-adaptive-plus` |
| Agentic RAG | tool-calling + retrieval-as-a-tool | `agentic-rag` |

The chat commands accept `--persist` to save memory to SQLite across runs.

## Setup

Install dependencies into a virtual environment:

```bash
uv sync
```

Configure the connection (copy and edit if your host differs):

```bash
copy .env.example .env
```

## Running

Every agent is its own console script. Run any of them with `uv run <command>`.

Chat + memory:

```bash
uv run chat-buffer
```

```bash
uv run chat-summary --persist
```

Tool calling:

```bash
uv run tools-react
```

```bash
uv run tools-native
```

RAG (one backend each):

```bash
uv run rag-numpy
```

```bash
uv run rag-chroma
```

```bash
uv run rag-llamaindex
```

```bash
uv run rag-llamaindex-chroma
```

```bash
uv run rag-haystack
```

```bash
uv run rag-rerank
```

Hybrid (chat + RAG):

```bash
uv run hybrid-semantic
```

```bash
uv run hybrid-llm
```

```bash
uv run hybrid-adaptive
```

```bash
uv run hybrid-adaptive-plus
```

Agentic RAG and evaluation:

```bash
uv run agentic-rag
```

```bash
uv run rag-eval
```

Common flags: `--no-soul` (all agents), `--persist` (chat), `--no-index` and `--drop` (RAG/hybrid/agentic).

### What each command does (in plain words)

- `chat-buffer` - a chatbot that remembers everything you said in this chat.
- `chat-summary` - a chatbot that remembers a short summary when the chat gets long.
- `tools-react` - a bot that can use tools (like a calculator) by writing them out as text.
- `tools-native` - the same idea, but it calls tools the "official" way the model supports.
- `rag-numpy` - answers questions by first reading your documents; the simplest version.
- `rag-chroma` - the same, but it saves what it read so it does not re-read every time.
- `rag-llamaindex` / `rag-haystack` - the same idea built with the LlamaIndex / Haystack libraries.
- `rag-llamaindex-chroma` - LlamaIndex doing the reading, with the saved store from Chroma.
- `rag-rerank` - a smarter search: it looks two ways and then re-sorts results to pick the best.
- `hybrid-semantic` / `hybrid-llm` - decides "should I read the docs or just chat?" before answering.
- `hybrid-adaptive` - reads the docs first, then checks "is this useful?" and chats if not.
- `hybrid-adaptive-plus` - the careful version: checks each document, the answer, and retries if needed.
- `agentic-rag` - the bot decides by itself when to search the documents, when to use a tool, or just answer. Searching the documents is given to it as one of its "tools".
- `rag-eval` - not a chat. It is a small test that scores how well each RAG version finds the right document, so you can compare them with numbers.

Inside the REPL: type a message, or use slash commands.

- `/skills` lists available skills (a numbered menu).
- `/<number>` or `/<name>` loads a skill into the active context.
- `/add [path]` ingests a text file into the RAG store live, without restarting.
  With no path (or an invalid one), it scans `data/corpus/` and adds any new files
  not yet indexed. Works for RAG and hybrid agents; persists for Chroma.
- `/win` prints the full context window currently sent to the model.
- `/active`, `/clear`, `/rules`, `/help`, `/exit`.

## MCP servers (extra tools)

The tool-calling agents (`tools-react`, `tools-native`) can use tools from any
Model Context Protocol server. Copy the example and edit it:

```bash
copy mcp.json.example mcp.json
```

Each entry is either a stdio server (`command` + `args`) or an HTTP server
(`url` + `transport`). On startup the agents connect, load the servers' tools,
and merge them with the built-ins (built-ins win on name clashes). If `mcp.json`
is absent or a server fails, the agents simply run with the built-in tools.

Set `MCP_CONFIG` to point at a different config file if needed.

## Skills and rules

- `SOUL.md` (project root) is a global persona/identity prepended to every agent's
  system prompt. Every agent accepts `--no-soul` to skip loading it.
- Agents can write durable facts into the managed `## Memory` section of `SOUL.md`
  (between `memory:start`/`memory:end` markers, append-only + deduped): use the
  `/remember <fact>` command in any agent, or the `remember` tool in the
  tool-calling agents. New memories apply on the next turn (no restart).
- `rules/` holds always-on markdown that is injected into every agent's system
  prompt automatically.
- `skills/` holds opt-in markdown skills loaded on demand via the `/` menu. Drop a
  new `.md` file in `skills/` (optionally with `name:`/`description:` frontmatter)
  and it appears in the menu.

## How RAG works

All four RAG backends implement one `Retriever` interface (`index` / `search`)
and return the same `RetrievedChunk(text, source, score)`, so the agents are
identical regardless of the engine behind them. The flow:

**At startup (index once):**

1. Load every file in `data/corpus/` and split it into chunks.
2. Send the chunks to Ollama `mxbai-embed-large` to get one vector per chunk.
3. Store the vectors in the backend (numpy matrix, Chroma, LlamaIndex, Haystack).

**Per query (search):**

4. Embed the query with the same model.
5. Score every chunk by cosine similarity and take the top-k (default 4).

**Answer (retrieve-then-read):**

6. If nothing is retrieved, the agent replies "I don't know".
7. Otherwise it builds `Context: [source] chunk... / Question: ...` and sends it
   to `gemma4:e4b` with a system rule: answer **only** from the context and cite
   sources as `[source]`.
8. Retrieval hits (sources + scores) and the LLM call are logged.

Two defining traits: the RAG-only agents are **stateless** (each query is
independent, no conversation memory), and embeddings come from a different model
(`mxbai-embed-large`) than generation (`gemma4:e4b`).

### The backends

- **pure-numpy** (`rag-numpy`): the most transparent. Chunks by character window
  with overlap, L2-normalizes vectors, and does a brute-force dot product
  (`matrix @ query_vec`) for cosine similarity. No database, O(N) per query - great
  for understanding, poor for scale.
- **pure-chroma** (`rag-chroma`): same idea but vectors live in a **persistent**
  Chroma collection (`.chroma/`) that performs the nearest-neighbor search. It is
  the only backend that survives restarts: if the collection already holds data it
  is reused instead of re-indexed.
- **llamaindex** (`rag-llamaindex`): uses LlamaIndex's `VectorStoreIndex` with a
  sentence splitter and Ollama embeddings/LLM, kept in memory.
- **llamaindex-chroma** (`rag-llamaindex-chroma`): same LlamaIndex orchestration,
  but with a **persistent Chroma** vector store behind it. This shows the common
  production layering - a framework owns the pipeline while an external, scalable
  store owns the vectors. Contrast it with `rag-chroma`, where our own code does
  the orchestration and Chroma is driven directly.
- **haystack** (`rag-haystack`): uses Haystack components (document splitter,
  in-memory store, embedding retriever) wired into the same interface.

Because the chunking differs (numpy/chroma split by characters; LlamaIndex and
Haystack split by sentences/words), the same question can retrieve slightly
different passages across backends - a good thing to compare.

- **rerank** (`rag-rerank`): a production-style retrieval stack on a small local
  scale. It runs **hybrid retrieval** - dense (Ollama embeddings, cosine) plus
  sparse (BM25 keyword) - fuses the two rankings with Reciprocal Rank Fusion, then
  **reranks** the candidates with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`
  by default) and keeps the top-k. The cross-encoder gives much sharper relevance
  separation than raw cosine (clearly positive scores for the right chunk, strongly
  negative for the rest). The model downloads on first run; configure it via
  `RERANK_MODEL` and the candidate pool via `FUSION_CANDIDATES`.

### RAG storage flags

Every RAG agent (and every hybrid agent) accepts:

- `--no-index` - start without indexing/uploading the corpus. The in-memory
  backends start empty; Chroma uses whatever is already persisted.
- `--drop` - drop persisted RAG storage before starting (Chroma only; a no-op for
  the in-memory backends), forcing a clean rebuild.

By default the in-memory backends index the corpus on every run, while Chroma
reuses its persisted collection if present.

```bash
uv run rag-chroma --drop
```

```bash
uv run rag-chroma --no-index
```

## Agentic RAG

`agentic-rag` is a different take on combining tools and retrieval: instead of a
router deciding RAG-vs-chat up front, retrieval is exposed to a native
tool-calling agent as a `search_knowledge_base` tool (backed by the hybrid
BM25+dense+rerank engine). The model decides per turn whether to search the
corpus, use another tool (calculator, shell, web_search, MCP), or answer
directly. It is instructed to prefer the knowledge base for factual questions.

```bash
uv run agentic-rag
```

## Production features

- **Human-in-the-loop**: dangerous tools (currently `shell`) require confirmation
  before running. Controlled by `REQUIRE_TOOL_APPROVAL` (default true); the user is
  prompted `run this? [y/N]` and can deny.
- **Reliability**: every LLM call retries on transient errors/timeouts
  (`LLM_MAX_RETRIES`) and generation/tool calls fall back to a second model
  (`FALLBACK_MODEL`, default `llama3.1:8b`) if the primary fails.
- **Observability (LangSmith)**: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`
  to trace every chain/LLM/tool step in LangSmith. No-op when disabled.
- **Evaluation**: `uv run rag-eval` runs a small labelled question set against the
  RAG backends and prints hit-rate and context-recall, for objective comparison and
  regression checks.

```bash
uv run rag-eval
```

## Logging

Each agent writes two channels:

- A readable, bordered console stream (no colors or emojis).
- A structured JSON-lines file at `logs/<agent>.jsonl` capturing LLM calls
  (prompt, response, latency, tokens), tool calls, retrieval hits with scores,
  routing decisions, and errors.

Tail a log to monitor an agent, for example:

```bash
Get-Content logs/hybrid-adaptive.jsonl -Wait
```

## Architecture

```
src/localagent/
  config.py          settings from .env (host, models, chunking, logging)
  llm.py             ChatOllama / OllamaEmbeddings factories + token usage
  logging_setup.py   dual console + JSON logging
  memory.py          buffer and summary-buffer memory with SQLite persistence
  skills.py          skill discovery for the / menu
  rules.py           always-on rules loader
  cli.py             REPL and slash-command engine
  tools/             example tools: calculator, shell, read_file, web_search (DuckDuckGo)
  mcp_tools.py       optional Model Context Protocol tool loading (mcp.json)
  rag/               common Retriever interface + numpy, chroma, llamaindex, haystack
  agents/            one module + console script per variant
```

LangChain orchestrates the agents, memory and routing. Each RAG backend
implements the same `Retriever` interface (`index` / `search`), so the agents do
not care which engine is behind it. The corpus in `data/corpus/` is indexed at
startup for the RAG and hybrid agents.
