# Memory and Tools

## Conversation memory

Memory lets an agent remember what was said earlier in a conversation.

- **Buffer memory** keeps every turn verbatim. It is simple and lossless but the
  context grows without bound and eventually exceeds the model's context window.
- **Summary-buffer memory** keeps the most recent turns verbatim and folds older
  turns into a running summary produced by the LLM. This bounds the context size
  while preserving the gist of the earlier conversation.

Memory can be ephemeral (lost when the process exits) or persistent (saved to a
store such as SQLite so it survives across runs).

## Tool calling

Tools are functions the agent can call to act on the world or compute something
the model is bad at, such as exact arithmetic.

- **ReAct / text protocol**: the model is instructed to emit a parseable action
  line, the runner executes the tool, and the result is fed back as an
  observation. This works with any model because it relies only on text.
- **Native tool calling**: the model returns a structured tool call (name and
  JSON arguments). The framework executes the tool and returns a tool message.
  This requires a model that supports tool calling.

A tool should have a clear name, a typed signature, and a concise description so
the model knows when and how to use it.
