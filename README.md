# RAG Bot

Enterprise-style retrieval augmented chatbot for simple `.txt` sources.

## What is included

- LangGraph-based chat pipeline
- FAISS vector store module
- Separate ingestion flow for text files
- PII masking on input, retrieval context, and output
- Guardrail nodes for inbound and outbound messages
- Phoenix-oriented tracing/evaluation hooks
- FastAPI service surface

## Layout

- `src/ragbot/ingestion/` for file ingestion and indexing
- `src/ragbot/vectorstore/` for FAISS storage
- `src/ragbot/safety/` for masking and guardrails
- `src/ragbot/graph/` for the chat workflow
- `src/ragbot/observability/` for Phoenix/trace hooks
- `src/ragbot/api/` for the FastAPI app

## Quick start

1. Create a Python 3.10+ environment or reuse the provided `.venv`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Put a `.txt` file in `data/raw/`.
4. Ingest it:

```bash
ragbot ingest --source data/raw/sample.txt
```

5. Run the API:

```bash
ragbot serve
```

6. Send a chat request to `POST /chat`.

## Environment variables

- `RAGBOT_DATA_DIR`
- `RAGBOT_INDEX_DIR`
- `RAGBOT_GEMINI_MODEL`
- `RAGBOT_GEMINI_EMBEDDING_MODEL`
- `GOOGLE_API_KEY`
- `PHOENIX_PROJECT_NAME`
- `PHOENIX_COLLECTOR_ENDPOINT`
