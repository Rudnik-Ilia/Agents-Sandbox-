# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) grounds an LLM's answers in an external
knowledge base. Instead of relying only on what the model memorized during
training, a RAG system retrieves relevant text at query time and includes it in
the prompt.

## The RAG pipeline

1. **Ingest**: load documents from files, databases, or the web.
2. **Chunk**: split documents into smaller passages so retrieval is precise.
3. **Embed**: convert each chunk into a vector using an embedding model. This
   project uses the `mxbai-embed-large` embedding model served by Ollama.
4. **Store**: keep the vectors in a vector store. Options range from a plain
   in-memory numpy matrix to dedicated databases such as Chroma.
5. **Retrieve**: embed the query and find the nearest chunks by cosine similarity.
6. **Generate**: pass the retrieved context plus the question to the LLM.

## Implementation choices

- **Pure / from scratch**: compute embeddings and cosine similarity directly with
  numpy. Most transparent and dependency-light.
- **Chroma**: a persistent local vector database that handles storage and
  approximate nearest-neighbor search.
- **LlamaIndex**: a data framework that provides loaders, node parsers, and a
  `VectorStoreIndex` with a high-level query engine.
- **Haystack**: a component-and-pipeline framework where embedders, retrievers,
  and generators are connected explicitly.

## Why chunking and overlap matter

Chunks that are too large dilute relevance and waste context; chunks that are too
small lose meaning. A small overlap between consecutive chunks preserves context
that would otherwise be cut at a boundary.
