# Evaluation System Documentation

## Overview

The RAG Chatbot includes a comprehensive evaluation system using Phoenix and LangGraph to assess chatbot responses across multiple quality dimensions:

- **Correctness**: Is the answer correct? (LLM-as-judge)
- **Relevance**: Is the response relevant to the query? (LLM-as-judge)
- **Groundedness**: Is the answer supported by retrieved context? (Hallucination detection)
- **Routing**: Did the retriever find relevant documents? (Deterministic validation)
- **Safety**: Is input/output safe from PII, injection, toxicity? (Hybrid: LLM + rules)
- **Latency**: Does response time meet SLA? (Threshold-based)
- **Format**: Does response conform to expected schema? (Pydantic validation)

## Architecture

```
User Request
    ↓
LangGraph Pipeline
    ├─ input_guardrails → [Safety evaluation]
    ├─ mask_input → [Format validation]
    ├─ retrieve → [Routing evaluation]
    ├─ generate → [Correctness, Relevance, Groundedness]
    ├─ output_guardrails → [Safety evaluation]
    ├─ mask_output → [Format validation]
    ↓
Phoenix Tracing (OpenTelemetry)
    ↓
Span-Level Evaluations
    ↓
Evaluation Results
    ├─ Online: Logged to Phoenix dashboard
    └─ Offline: Saved to JSON/CSV for analysis
```

## Evaluators

### 1. Correctness Evaluator

**Type**: LLM-based  
**SDK**: Phoenix `ClassificationEvaluator`  
**Applied to**: `generate` node  
**Input**: User question, bot response  
**Output**: "correct" or "incorrect"

**Usage**:
```python
from ragbot.evaluations.correctness import CorrectnessEvaluator

evaluator = CorrectnessEvaluator(enabled=True)
result = await evaluator.evaluate({
    "question": "What is the capital of France?",
    "response": "The capital of France is Paris."
})
# result.score: 1.0, result.passed: True
```

### 2. Relevance Evaluator

**Type**: LLM-based  
**SDK**: Phoenix `ClassificationEvaluator`  
**Applied to**: `generate` node  
**Input**: User question, bot response  
**Output**: "relevant" or "irrelevant"

Determines if the response addresses the user's question topic.

### 3. Groundedness Evaluator

**Type**: LLM-based  
**SDK**: Phoenix `FaithfulnessEvaluator`  
**Applied to**: `generate` node  
**Input**: Question, response, retrieved documents, context  
**Output**: Faithfulness score (0-1), Hallucination detection

Detects if response claims are supported by retrieved context.

**Usage**:
```python
from ragbot.evaluations.groundedness import GroundednessEvaluator

evaluator = GroundednessEvaluator(enabled=True)
result = await evaluator.evaluate({
    "question": "What color is the sky?",
    "response": "The sky is blue.",
    "contexts": ["The sky appears blue due to Rayleigh scattering."],
})
# Faithfulness score indicates hallucination risk
```

### 4. Routing Evaluator

**Type**: Deterministic (rule-based)  
**Applied to**: `retrieve` node  
**Input**: Retrieved documents, similarity scores  
**Output**: Score, validation status

**Rules**:
- Minimum documents retrieved: ≥ 1
- Minimum similarity score: ≥ 0.3
- Average score is reported

**Usage**:
```python
from ragbot.evaluations.routing import RoutingEvaluator

evaluator = RoutingEvaluator(min_doc_count=1, min_score_threshold=0.3)
result = await evaluator.evaluate({
    "retrieved_docs": ["doc1", "doc2"],
    "scores": [0.8, 0.6],
})
# result.passed: True if documents retrieved with good scores
```

### 5. Safety Evaluator

**Type**: Hybrid (LLM + rules)  
**Applied to**: `input_guardrails`, `output_guardrails` nodes  
**Input**: Text to evaluate  
**Output**: "safe" or "unsafe", violation list

**Deterministic checks**:
- PII detection (email, phone, SSN, credit card)
- Prompt injection patterns (e.g., "ignore previous instructions")
- Harmful keywords (e.g., "malware", "exploit")

**LLM-based judgment**: Additional safety classification

**Usage**:
```python
from ragbot.evaluations.safety import SafetyEvaluator

evaluator = SafetyEvaluator(
    check_pii=True,
    check_injection=True,
    check_profanity=True
)
result = await evaluator.evaluate({
    "text": "Contact support at support@example.com",
    "context": "output"
})
# result.passed: False (PII detected)
```

### 6. Latency Evaluator

**Type**: Deterministic (threshold-based)  
**Applied to**: All spans  
**Input**: Duration in milliseconds  
**Output**: "pass", "warning", or "fail"

**Thresholds**:
- **Pass**: < 3 seconds (score: 1.0)
- **Warning**: 3-5 seconds (score: 0.75)
- **Fail**: > 5 seconds (score: 0.0)

**Usage**:
```python
from ragbot.evaluations.latency import LatencyEvaluator

evaluator = LatencyEvaluator(
    pass_threshold_ms=3000,
    warning_threshold_ms=5000
)
result = await evaluator.evaluate({
    "duration_ms": 2500,
    "node_name": "generate"
})
# result.passed: True, result.metadata["status"]: "pass"
```

### 7. Format Evaluator

**Type**: Deterministic (schema validation)  
**Applied to**: `generate`, `output_guardrails` nodes  
**Input**: Response (JSON or dict)  
**Output**: Validation status, schema errors

**Validation**:
- JSON structure validity
- Pydantic schema compliance
- Required fields present
- Type correctness

**Usage**:
```python
from ragbot.evaluations.format import FormatEvaluator
from ragbot.schemas import ChatResponse

evaluator = FormatEvaluator(schema=ChatResponse)
result = await evaluator.evaluate({
    "response": {
        "answer": "Test",
        "sources": [],
        "input_safety": {"allowed": True, "reason": "safe"},
        "output_safety": {"allowed": True, "reason": "safe"}
    }
})
# result.passed: True if valid ChatResponse
```

## Using the Evaluation Runner

The `EvaluationRunner` orchestrates all evaluators:

```python
from ragbot.evaluations.runner import EvaluationRunner

# Create runner with default evaluators
runner = EvaluationRunner()

# Evaluate a single request
result = await runner.evaluate(
    input_text="What is X?",
    response_text="X is a variable.",
    question="What is X?",
    context="Relevant context here...",
    retrieved_docs=["doc1", "doc2"],
    duration_ms=2500,
)

# Access results
for evaluator_name, eval_result in result.evaluations.items():
    print(f"{evaluator_name}: {eval_result.score} ({eval_result.passed})")

# Get aggregate metrics
print(f"Aggregate score: {result.aggregate_score()}")
print(f"Pass rate: {result.pass_rate()}")

# Batch evaluation
requests = [
    {
        "input_text": "Q1?",
        "response_text": "A1.",
        "duration_ms": 1500,
    },
    {
        "input_text": "Q2?",
        "response_text": "A2.",
        "duration_ms": 2000,
    },
]
batch_results = await runner.evaluate_batch(requests)
```

## Offline Evaluation

Run batch evaluation on test datasets:

```python
from ragbot.evaluations.offline import OfflineEvaluationPipeline

pipeline = OfflineEvaluationPipeline()

# Load test cases from JSON
test_cases = pipeline.load_test_cases("data/eval/test_cases.json")

# Run and save results
results = await pipeline.run_and_save(
    test_cases=test_cases,
    output_dir="data/eval/results",
    run_name="Evaluation Run 1",
)

# Access summary
summary = results.to_summary()
print(f"Mean aggregate score: {summary['mean_aggregate_score']}")
print(f"Evaluator statistics: {summary['evaluators']}")
```

### Output Files

Results are saved as:
- `{run_id}_results.json` - Full trace results
- `{run_id}_results.csv` - Tabular format for analysis
- `evaluation_report.json` - Summary statistics

### Offline Evaluation Command

```bash
python -m ragbot.evaluations.offline \
    --test-dataset data/eval/test_cases.json \
    --output-dir data/eval/results \
    --run-name "My Evaluation"
```

## Online Evaluation (Production)

To enable online evaluation during chatbot inference:

```python
from ragbot.evaluations.runner import EvaluationRunner
from ragbot.service import ChatService

# Create service with evaluators enabled
service = ChatService.create(enable_evals=True)

# Evaluators run automatically on each request
response = service.chat("What is X?")

# Results logged to Phoenix dashboard
# Disabled by default in production to minimize latency
```

**Configuration**:
```python
from ragbot.evaluations.config import EvaluationConfig

config = EvaluationConfig(
    enabled=True,
    online_enabled=True,
    online_sample_rate=0.1,  # Evaluate 10% of requests
    offline_enabled=True,
)
```

## Integration with Chat Graph

Evaluators are integrated into the LangGraph pipeline:

```python
from ragbot.graph.chat_graph import ChatRuntime, build_chat_app
from ragbot.evaluations.runner import EvaluationRunner

# Create runtime with evaluator support
runtime = ChatRuntime(
    store=faiss_store,
    answerer=gemini_answerer,
    pii_masker=pii_masker,
    guardrails=guardrails,
    evaluator_runner=EvaluationRunner(),  # Enable evaluations
)

# Build graph with evaluation hooks
app = build_chat_app(runtime)

# Evaluation results logged to OpenTelemetry spans
response = app.invoke({
    "message": "What is X?",
    "top_k": 4
})
```

**Evaluation logging in spans**:
```
ragbot.eval.generate.correctness.score: 0.95
ragbot.eval.generate.correctness.passed: true
ragbot.eval.generate.relevance.score: 0.87
ragbot.eval.generate.groundedness.score: 0.92
ragbot.eval.generate.aggregate_score: 0.91
ragbot.eval.retrieve.routing.score: 1.0
ragbot.eval.retrieve.routing.passed: true
ragbot.eval.input_guardrails.safety.score: 1.0
ragbot.eval.input_guardrails.safety.passed: true
```

## Phoenix Dashboard Integration

When evaluators are enabled, results appear in Phoenix:

1. **Spans view**: Filter by evaluation span attributes
2. **Traces tab**: View full trace with evaluation metrics
3. **Datasets**: Compare evaluation scores across versions
4. **Experiments**: Run A/B tests with evaluation metrics

## Testing Evaluators

Run unit tests:

```bash
# Test deterministic evaluators
pytest tests/eval/test_deterministic_evaluators.py -v

# Test evaluation runner
pytest tests/eval/test_runner.py -v

# Test offline pipeline
pytest tests/eval/test_offline.py -v

# All evaluation tests
pytest tests/eval/ -v
```

## Configuration

Configure evaluators via environment variables or `EvaluationConfig`:

```python
from ragbot.evaluations.config import EvaluationConfig

config = EvaluationConfig(
    enabled=True,
    online_enabled=True,
    online_sample_rate=0.1,  # 10% of requests
    offline_enabled=True,
    llm_model="gemini-1.5-flash",
    llm_temperature=0.0,  # Deterministic
    latency_thresholds={
        "pass": 3000,    # milliseconds
        "warning": 5000,
        "fail": float("inf"),
    },
    safety_check_pii=True,
    safety_check_injection=True,
    safety_check_profanity=True,
    format_strict_validation=True,
)
```

## Troubleshooting

### Evaluators Not Running

- Check `enable_evals=True` in ChatService
- Verify evaluator has `enabled=True`
- Check OpenTelemetry tracing is initialized

### LLM Evaluator Failures

- Verify Google API key is set for Gemini
- Check network connectivity to Gemini API
- Fallback to deterministic checks if LLM unavailable

### Low Evaluation Scores

- Check if responses are actually correct (use manual review)
- Adjust LLM eval prompts in `config.py`
- Enable per-evaluator debugging with detailed metadata

## Best Practices

1. **Online vs Offline**:
   - Use offline evaluation for comprehensive testing (100% eval rate)
   - Use online evaluation for production monitoring (sample rate: 5-10%)

2. **LLM Choose**:
   - Gemini (default): Fast, cost-effective, consistent with main pipeline
   - GPT-4: Higher quality but slower and more expensive

3. **Evaluation Frequency**:
   - Run offline eval after model updates
   - Monitor online eval metrics in Phoenix dashboard
   - Set up alerts for low evaluation scores

4. **Dataset Management**:
   - Create hand-curated golden dataset of known good answers
   - Include edge cases and ambiguous questions
   - Versioning: keep historical test datasets for regression analysis

## Related Documentation

- [Phoenix documentation](https://docs.arize.com/phoenix)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [OpenInference specification](https://github.com/Arize-ai/openinference)
