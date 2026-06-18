# RAG Bot

Enterprise-style retrieval augmented chatbot for simple `.txt` sources.

## Quick Start

**New to this project?** Start here: [QUICKSTART.md](QUICKSTART.md)

The fastest way to get running:

```bash
# Activate environment
source .venv/bin/activate

# Check dependencies
python check_deps.py

# Start the server
python main.py serve

# In another terminal, test it
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is in the data?"}'
```


Visit http://localhost:8000/docs for interactive API documentation.

- LangGraph-based chat pipeline with explicit node routing
- FAISS vector store module with fallback numpy backend
- Separate ingestion flow for text files with PII masking
- PII masking on input, retrieval context, and output
- Guardrail nodes for inbound and outbound messages
- **Phoenix auto-instrumentation** for automatic tracing (no manual spans!)
- FastAPI service surface with Swagger UI documentation

## Layout

- `src/ragbot/ingestion/` — Text file ingestion and chunking pipeline
- `src/ragbot/vectorstore/` — FAISS vector storage with persistence
- `src/ragbot/safety/` — PII masking and guardrails nodes
- `src/ragbot/graph/` — LangGraph chat workflow
- `src/ragbot/observability/` — Phoenix tracing hooks and bootstrap
- `src/ragbot/api/` — FastAPI application entrypoints
- `data/` — Data directory (raw inputs and FAISS index)
- `tests/` — Unit tests for core components

## Quick start: Setup

### 1. Activate the virtual environment

```bash
source .venv/bin/activate
```

Or reference the virtualenv directly in commands:

```bash
.venv/bin/python -m pip list
```

### 2. Install dependencies

Install from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or use `pyproject.toml` for editable install:

```bash
pip install -e .
pip install -e ".[dev]"
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
# Google Gemini API key (required)
GOOGLE_API_KEY=your_google_api_key_here

# Phoenix tracing project name and collector endpoint
PHOENIX_PROJECT_NAME=test-bot-with-eval
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

If running without `.env`, export variables in your shell:

```bash
export GOOGLE_API_KEY="your_key_here"
export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"
```

## Quick start: Usage

### Ingest a text file

Place a `.txt` file in `data/raw/`, then ingest it:

```bash
ragbot ingest --source data/raw/sample.txt
```

Or directly (without CLI):

```bash
.venv/bin/python -c "
from ragbot.service import ChatService
service = ChatService.create()
result = service.ingestion.ingest_file('data/raw/sample.txt')
print(result.model_dump_json(indent=2))
"
```

### Run the API server

Start the FastAPI service:

```bash
ragbot serve
```

Or with custom host/port:

```bash
ragbot serve --host 0.0.0.0 --port 8080
```

Or via environment variables:

```bash
RAGBOT_HOST=0.0.0.0 RAGBOT_PORT=8080 ragbot serve
```

The server will print endpoints:

```
Starting RAG Bot API server on http://127.0.0.1:8000
  POST /chat — Send a message for the chatbot to answer
  POST /ingest — Ingest a text file into the index
  GET /health — Health check endpoint
  GET /status — Service status and index information
  GET /docs — Interactive API documentation (Swagger UI)
  GET /redoc — ReDoc API documentation
```

### Fetch traces and spans from Phoenix

Fetch CHAIN spans for a fixed time window:

```bash
ragbot fetch-phoenix \
  --from 2026-06-01T09:00:00Z \
  --to 2026-06-01T10:00:00Z \
  --span-kind CHAIN
```

Run in incremental mode using a local checkpoint:

```bash
ragbot fetch-phoenix --delta --to 2026-06-01T10:00:00Z
```

Optional loop mode for worker-style execution:

```bash
ragbot fetch-phoenix --delta --poll-interval-seconds 60
```

### API Endpoints

#### Health check

```bash
curl http://localhost:8000/health
```

#### Service status

```bash
curl http://localhost:8000/status
```

#### Ingest a file

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "data/raw/sample.txt",
    "rebuild": true
  }'
```

#### Send a chat message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the emergency hotline?",
    "conversation_id": "user_123",
    "top_k": 4
  }'
```

Response:

```json
{
  "answer": "The emergency hotline is [PHONE_1].",
  "sources": [
    {
      "chunk_id": "chunk_abc123",
      "source_path": "data/raw/sample.txt",
      "text": "emergency hotline is 555-123-4567",
      "score": 0.95,
      "chunk_index": 1,
      "metadata": {"was_masked": true, "pii_entities": [...]}
    }
  ],
  "input_safety": {
    "allowed": true,
    "reason": "Input passed guardrails.",
    "warnings": [],
    "masked_text": "What is the emergency [PHONE_1]?"
  },
  "output_safety": {
    "allowed": true,
    "reason": "Output passed guardrails.",
    "warnings": [],
    "masked_text": "The emergency hotline is [PHONE_1]."
  },
  "trace_id": "trace_xyz789",
  "conversation_id": "user_123"
}
```

#### Interactive docs

Open http://localhost:8000/docs in your browser to explore endpoints interactively.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (none) | Google Generative AI API key (required for Gemini) |
| `RAGBOT_EMBEDDING_PROVIDER` | `sentence-transformer` | Embedding backend (`sentence-transformer`, `gemini`, or `hash`) |
| `RAGBOT_SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformer embedding model |
| `RAGBOT_GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model to use for generation |
| `RAGBOT_GEMINI_EMBEDDING_MODEL` | `models/text-embedding-004` | Gemini embedding model |
| `RAGBOT_DATA_DIR` | `./data` | Root data directory |
| `RAGBOT_RAW_DIR` | `./data/raw` | Directory for raw input files |
| `RAGBOT_INDEX_DIR` | `./data/index` | Directory for FAISS index and metadata |
| `RAGBOT_MAX_INPUT_CHARS` | `6000` | Maximum input message length |
| `RAGBOT_RETRIEVAL_TOP_K` | `4` | Number of retrieval results to use |
| `PHOENIX_PROJECT_NAME` | `test-bot-with-eval` | Phoenix project name for tracing |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | Phoenix collector endpoint URL |
| `PHOENIX_QUERY_ENDPOINT` | `http://localhost:6006` | Phoenix query endpoint used by fetch worker |
| `PHOENIX_FETCH_OUTPUT_DIR` | `data/fetch/phoenix` | Output directory for fetched JSONL files |
| `PHOENIX_FETCH_CHECKPOINT_FILE` | `data/fetch/phoenix/checkpoint.json` | Local incremental checkpoint file |
| `PHOENIX_FETCH_BATCH_SIZE` | `200` | Max traces/spans fetched per adapter call |
| `PHOENIX_FETCH_SPAN_KIND` | `CHAIN` | Default span kind filter for fetch worker |
| `BATCH_EVALUATION_CRON_ENABLED` | `true` | Start the hourly batch evaluation scheduler with the API server |
| `BATCH_EVALUATION_CRON_INTERVAL_SECONDS` | `3600` | Batch evaluation scheduler interval |
| `BATCH_EVALUATION_CRON_LIMIT` | `1000` | Max Phoenix spans fetched per scheduled batch |
| `BATCH_EVALUATION_CRON_SYNC_ANNOTATIONS` | `true` | Wait for scheduled annotation writes to sync |
| `BATCH_EVALUATION_CRON_SAVE_ANNOTATIONS` | `true` | Write scheduled evaluation annotations back to Phoenix |
| `BATCH_EVALUATION_USE_CHECKPOINT` | `false` | Enable exact-watermark checkpointing for batch evaluation |
| `BATCH_EVALUATION_CHECKPOINT_FILE` | `data/batch_evaluation/checkpoint.json` | Local batch evaluation checkpoint file |
| `RAGBOT_HOST` | `127.0.0.1` | API server host |
| `RAGBOT_PORT` | `8000` | API server port |
| `RAGBOT_RELOAD` | `false` | Enable hot-reload (dev only) |

When the API server starts, the batch evaluation scheduler runs on a daemon thread. It waits one interval, then evaluates `PHOENIX_FETCH_SPAN_KIND` spans and saves annotations to Phoenix.

## Batch evaluation pipeline

The batch evaluation job reads traced LangGraph application spans from Phoenix, scores them with the local evaluators already used by the project, enriches the results with RCA metadata, and writes the final annotations back to Phoenix.

### End-to-end flow

1. `resolve_batch_evaluation_config(...)` normalizes the run window.
   If `BATCH_EVALUATION_USE_CHECKPOINT=true`, the job loads `BATCH_EVALUATION_CHECKPOINT_FILE`.
   When the checkpoint has `high_watermark_time`, the job fetches from that exact timestamp forward.
   When the checkpoint is missing or empty, the job preserves an explicit `from_time` if provided; otherwise it backfills Phoenix history by leaving `from_time=None`.
2. The batch evaluator requests only root spans from Phoenix by forcing `root_spans_only=True`, then keeps only root `CHAIN` spans whose main application node is `LangGraph`.
3. The evaluator fetches existing annotations only for those filtered root `LangGraph` spans to decide whether a span is already fully evaluated.
4. Root spans are skipped only if they already have the full batch annotation set:
   `correctness`, `relevance`, `faithfulness`, and `safety`.
   Root spans with partial annotations are re-evaluated.
   Child spans in the same trace do not affect this eligibility check.
5. `filter_checkpointed_spans(...)` removes only spans that match the exact checkpoint watermark tie-set.
   The composite key format is:
   `context.trace_id|start_time|name`
   Fallbacks are `trace_id` for the trace component and `span_name` for the name component.
6. `_build_evaluation_frame(...)` derives the evaluator input fields:
   `input`, `output`, `reference`, `span_id`, and `span_kind`.
7. `EvaluationRunner.evaluate_dataframe(...)` runs the four evaluators:
   `correctness`, `relevance`, `groundedness`, and `safety`.
8. `_build_annotations_frame(...)` converts evaluator outputs into Phoenix annotation rows.
   `groundedness` is written back under the Phoenix annotation name `faithfulness`.
9. `build_rca_feature_frame(...)` constructs the RCA feature frame from spans plus annotations.
10. `generate_rca(...)` produces the final RCA results.
11. `merge_final_rca_into_annotations(...)` adds selected RCA fields into annotation metadata.
12. `log_phoenix_span_annotations(...)` writes the merged annotations back to Phoenix.
13. Only after a successful Phoenix write, `update_checkpoint_after_success(...)` advances the local batch-evaluation checkpoint.

### Checkpoint behavior

Batch evaluation now uses a JSON checkpoint instead of timestamp-only checkpointing. The file stores:

```json
{
  "high_watermark_time": "2026-06-17T10:30:00+00:00",
  "processed_keys": [
    "trace_id|start_time|span_name"
  ]
}
```

The checkpoint is exact-watermark based:

- `high_watermark_time` is the maximum `start_time` from spans successfully evaluated and written back in the run.
- `processed_keys` stores only the composite keys whose `start_time` exactly matches the current `high_watermark_time`.
- Those keys are used only to avoid reprocessing duplicate ties when Phoenix returns spans at the watermark timestamp again on the next run.
- If evaluation fails or Phoenix annotation write-back fails, the checkpoint is not advanced.
- If `BATCH_EVALUATION_CRON_SAVE_ANNOTATIONS=false`, the job still evaluates spans but does not advance the checkpoint because no successful write-back occurred.

This design allows a full-history backfill when no checkpoint exists, while still preventing duplicate reprocessing when multiple spans share the exact checkpoint timestamp.

## Testing

Run the included tests:

```bash
pytest tests/
```

Or test specific modules:

```bash
pytest tests/test_pii.py tests/test_guardrails.py tests/test_faiss_store.py
```

## Architecture

### Chat Flow (LangGraph)

```
User Input
    ↓
Input Guardrails ← Checks for injection, length
    ↓
PII Mask Input ← Redacts emails, phones, SSNs
    ↓
Retrieve ← FAISS similarity search
    ↓
Generate ← Gemini with context
    ↓
Output Guardrails ← Validates model output
    ↓
PII Mask Output ← Redacts leakage
    ↓
Response (with trace_id, safety decisions)
```

### Component Interaction

- **Ingestion**: Read → Chunk → Mask PII → Embed → Index (FAISS)
- **Chat**: Guard → Mask → Retrieve → Generate → Guard → Mask
- **Observability**: Wrap all steps with OpenTelemetry spans → Phoenix
- **API**: FastAPI routes map to service methods

## Development

### Adding a new PII entity type

Edit `src/ragbot/safety/pii.py`:

```python
class PIIMasker:
    def __init__(self) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            ("email", re.compile(r"...")),
            ("your_new_type", re.compile(r"...")),  # Add here
        ]
```

### Adding a new guardrail

Edit `src/ragbot/safety/guardrails.py`:

```python
def check_input(self, text: str) -> GuardrailDecision:
    # Add your check here
    if some_condition(text):
        return GuardrailDecision(False, "Reason for block")
    return GuardrailDecision(True, "Passed")
```

### Swapping the LLM provider

Edit `src/ragbot/llm/gemini.py` or add a new module, then update `src/ragbot/graph/chat_graph.py` to wire it:

```python
answerer=YourNewAnswerer(model_name, api_key)
```

### Swapping the embedding provider

If not using Gemini embeddings, use the fallback `HashEmbeddingProvider` or implement your own in `src/ragbot/embeddings/providers.py`, then inject it in `src/ragbot/service.py`.

## Observability & Evaluation

Every chat request is automatically traced using **OpenTelemetry + Arize Phoenix** with **auto-instrumentation**. This means Phoenix automatically instruments:
- FastAPI HTTP routes
- LangChain components
- Business logic (guardrails, retrieval, generation)
- External APIs (Gemini)

**No manual span code needed!** Just write clean business logic and Phoenix automatically creates detailed traces.

Resulting trace structure:
```
POST /chat [500ms]
├─ starlette.request [500ms]
├─ guardrails.check_input [2ms]
├─ faiss_store.search [50ms]
├─ gemini.invoke [410ms]
└─ output validation [1ms]
```

### Quick Start with Phoenix

Launch a local Phoenix instance:

```bash
pip install arize-phoenix
phoenix launch
```

Then point your chatbot to it:

```bash
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
ragbot serve
```

Send a chat request and view the trace in the Phoenix UI at http://localhost:6006.

### Documentation

- **[AUTO_INSTRUMENTATION.md](AUTO_INSTRUMENTATION.md)** — How auto-instrumentation works (new!)
- **[PHOENIX.md](PHOENIX.md)** — Complete setup guide (local, Docker, managed cloud)
- **[TRACING.md](TRACING.md)** — Quick reference for spans, attributes, and code examples

Key topics:
- How Phoenix tracing works (OTel SDK → Phoenix collector)
- Viewing and analyzing traces in Phoenix UI
- Recording evaluations with `evaluations.jsonl`
- Performance considerations and production deployment
- Troubleshooting missing traces

## Production Considerations

1. **PII Masking Strategy**: The system masks PII before indexing and again on input/output. Tune the patterns in `pii.py` for your domain.
2. **Guardrails Strictness**: Adjust `guardrails.py` thresholds based on your risk tolerance. See `.env` for `RAGBOT_MAX_INPUT_CHARS`.
3. **Retrieval Quality**: Increase `RAGBOT_RETRIEVAL_TOP_K` if answers lack context; decrease if irrelevant results appear.
4. **Embedding Model**: For production, replace `HashEmbeddingProvider` with a real embedding service (Gemini, OpenAI, Hugging Face).
5. **LLM Model**: `gemini-1.5-flash` is fast but less capable; use `gemini-1.5-pro` for complex reasoning.
6. **Index Persistence**: FAISS index is persisted in `RAGBOT_INDEX_DIR`. Back this directory for durability.
7. **Concurrency**: FastAPI/Uvicorn handles concurrent requests; tune `--workers` based on load testing.
8. **Logging**: Use environment variables to control verbosity in production.

## License

MIT
