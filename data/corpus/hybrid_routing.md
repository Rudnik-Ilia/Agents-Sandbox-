# Hybrid Routing: Choosing RAG vs the LLM

A hybrid agent answers small talk directly with the LLM but switches to retrieval
when a question needs grounded knowledge. The interesting design question is how
to decide. Several modern approaches exist.

## Semantic router

Embed the incoming query and compare it, by cosine similarity, to pre-embedded
exemplars that represent "retrieval" topics versus "chit-chat". Pick the closer
group. This is fast and needs no extra LLM call, but it depends on good exemplars.

## LLM classifier router

Ask the LLM itself to classify the query, ideally with structured output (JSON or
a tool call) constrained to a fixed set of routes. This is flexible and easy to
explain, at the cost of one extra model call per query.

## Adaptive RAG

Always retrieve first, then have the LLM grade whether the retrieved documents are
actually relevant. If they are, generate an answer from them; if not, fall back to
plain chat. This is robust because the routing decision is informed by what was
actually retrieved. It is commonly implemented as a small state graph, for example
with LangGraph: a retrieve node, a grading edge, and generate or fallback nodes.

## Trade-offs

- Semantic routing is cheapest but least context-aware.
- LLM classification is a good balance of flexibility and cost.
- Adaptive RAG is the most reliable but does the most work per query.
