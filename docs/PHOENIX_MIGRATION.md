# Phoenix SDK Refactoring: Migration Guide

## Overview

The evaluation system has been completely refactored to use **Phoenix SDK native APIs** instead of custom evaluator classes. This provides:

- ✅ Direct integration with Phoenix's proven evaluation framework
- ✅ LLM-as-judge capabilities with proper classification
- ✅ Built-in evaluators (FaithfulnessEvaluator, DocumentRelevanceEvaluator)
- ✅ Efficient batch evaluation via `evaluate_dataframe()`
- ✅ Consistent pattern across all evaluators: `create_evaluator()` → Phoenix Evaluator

---

## What Changed

### Evaluator Classes → Factory Functions

**Before:**
```python
from ragbot.evaluations import CorrectnessEvaluator

evaluator = CorrectnessEvaluator(enabled=True)
result = await evaluator.evaluate({"question": "...", "response": "..."})
```

**After:**
```python
from ragbot.evaluations.correctness import create_correctness_evaluator
import pandas as pd

evaluator = create_correctness_evaluator()
df = pd.DataFrame([{"input": "...", "output": "..."}])
results_df = await evaluator.apply_async(df)  # Phoenix API
```

### Runner API

**Before:**
```python
from ragbot.evaluations import EvaluationRunner

runner = EvaluationRunner()
result = await runner.evaluate(
    input_text="Question",
    response_text="Answer"
)
# result: TraceEvaluationResults with custom EvaluationScore objects
```

**After:**
```python
from ragbot.evaluations import EvaluationRunner

runner = EvaluationRunner()
result = await runner.evaluate(
    input_text="Question", 
    response_text="Answer"
)
# Still returns TraceEvaluationResults (backwards compatible!)
```

### Batch Evaluation

**New Phoenix API Support:**
```python
import pandas as pd
from ragbot.evaluations import EvaluationRunner

runner = EvaluationRunner()
df = pd.DataFrame({
    "input": ["Q1", "Q2"],
    "output": ["A1", "A2"],
})

results_df = await runner.evaluate_dataframe(df)
# Phoenix evaluators add columns: correctness, relevance, etc.
```

---

## Evaluator Mapping

| Evaluator | Type | Implementation | Phoenix API |
|-----------|------|------------------|-------------|
| **Correctness** | LLM-as-judge | `create_classifier()` | classification template |
| **Relevance** | LLM-as-judge | `create_classifier()` | classification template |
| **Groundedness** | Built-in | `FaithfulnessEvaluator()` | faithfulness scoring |
| **Routing** | Deterministic | `create_evaluator()` | function-based validation |
| **Safety** | Hybrid | `create_classifier()` + regex | classification + patterns |
| **Latency** | Deterministic | `create_evaluator()` | threshold logic |
| **Format** | Deterministic | `create_evaluator()` | Pydantic validation |

---

## Factory Functions Reference

### Correctness
```python
from ragbot.evaluations.correctness import create_correctness_evaluator

evaluator = create_correctness_evaluator(llm=None)  
# llm: Optional Phoenix LLM instance (uses default if None)
```

**Template Pattern:**
```
Question: {input}
Response: {output}

Classify: CORRECT or INCORRECT
```

### Relevance
```python
from ragbot.evaluations.relevance import create_relevance_evaluator

evaluator = create_relevance_evaluator(llm=None)
```

**Template Pattern:**
```
Question: {input}
Response: {output}

Classify: RELEVANT or IRRELEVANT
```

### Groundedness
```python
from ragbot.evaluations.groundedness import create_groundedness_evaluator

evaluator = create_groundedness_evaluator(llm=None)
# Uses Phoenix's built-in FaithfulnessEvaluator
```

**Requires:** input, output, reference (context)

### Routing
```python
from ragbot.evaluations.routing import create_routing_evaluator

evaluator = create_routing_evaluator(
    min_doc_count=1,
    min_score_threshold=0.3
)
```

**Function Returns:** "PASSING" or "FAILING"

### Safety
```python
from ragbot.evaluations.safety import create_safety_evaluator

evaluator = create_safety_evaluator(
    llm=None,
    check_pii=True,
    check_injection=True
)
```

**Combines:** LLM classification + PII/injection regex patterns

### Latency
```python
from ragbot.evaluations.latency import create_latency_evaluator

evaluator = create_latency_evaluator(
    pass_threshold_ms=3000,
    warning_threshold_ms=5000
)
```

**Function Returns:** "PASS", "WARNING", or "FAIL"

### Format
```python
from ragbot.evaluations.format import create_format_evaluator
from pydantic import BaseModel

class ChatResponse(BaseModel):
    response: str

evaluator = create_format_evaluator(schema=ChatResponse)
```

**Function Returns:** "VALID" or "INVALID"

---

## LLM Configuration

Phoenix evaluators need credentials configured:

### Google Gemini (Default)
```bash
export GOOGLE_API_KEY="your-api-key"
```

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
```

### Custom LLM
```python
from ragbot.evaluations.config import get_phoenix_llm
from ragbot.evaluations.correctness import create_correctness_evaluator

custom_llm = get_phoenix_llm()  # Uses env-based provider config
evaluator = create_correctness_evaluator(llm=custom_llm)
```

---

## Results Schema

### TraceEvaluationResults (Backwards Compatible!)
```python
result = await runner.evaluate(...)

# Available methods:
result.aggregate_score()  # Mean of all evaluator scores (0-1)
result.pass_rate()        # Percentage of evaluators that passed
result.to_dict()          # Convert to dictionary
```

### EvaluationResult
```python
for evaluator_name, eval_result in result.evaluations.items():
    print(f"{evaluator_name}:")
    print(f"  score: {eval_result.score}")     # 0-1 float
    print(f"  passed: {eval_result.passed}")   # bool
    print(f"  reason: {eval_result.reason}")   # str

# Factory for Phoenix Score objects:
from ragbot.evaluations.results import EvaluationResult

phoenix_score = ...  # From Phoenix evaluator
eval_result = EvaluationResult.from_phoenix_score(
    "correctness", 
    phoenix_score
)
```

---

## Batch Evaluation Example

```python
import pandas as pd
from ragbot.evaluations import EvaluationRunner

# Create evaluator runner
runner = EvaluationRunner()

# Prepare data
requests = [
    {
        "input_text": "What is Python?",
        "response_text": "Python is a programming language.",
        "question": "What is Python?"
    },
    {
        "input_text": "What is JavaScript?",
        "response_text": "JavaScript is used for web development.",
        "question": "What is JavaScript?"
    }
]

# Evaluate batch
results = await runner.evaluate_batch(requests)

# Process results
for trace_result in results:
    print(f"Trace: {trace_result.trace_id}")
    print(f"Aggregate Score: {trace_result.aggregate_score():.2f}")
    print(f"Pass Rate: {trace_result.pass_rate():.0%}")
    for eval_name, eval_result in trace_result.evaluations.items():
        print(f"  {eval_name}: {eval_result.score:.2f} ({eval_result.reason})")
```

---

## Offline Evaluation Example

```python
from ragbot.evaluations.offline import OfflineEvaluationPipeline

pipeline = OfflineEvaluationPipeline()

# Load test cases
test_cases = pipeline.load_test_cases("data/eval/test_cases.json")

# Run and save evaluation
results = await pipeline.run_and_save(
    test_cases=test_cases,
    output_dir="results/eval_run_001",
    run_name="Model v1.0 Evaluation"
)

# Results saved as:
# - results/eval_run_001/run_*.json (detailed results)
# - results/eval_run_001/run_*.csv (CSV export)
# - results/eval_run_001/evaluation_report.json (summary stats)
```

---

## Chat Graph Integration

No changes needed! The existing `evaluate_step()` method works seamlessly:

```python
runtime = ChatRuntime()
runtime.evaluator_runner = EvaluationRunner()  # New runner works as-is

# Evaluations are logged as span attributes:
# ragbot.eval.{step_name}.{evaluator}.score
# ragbot.eval.{step_name}.{evaluator}.passed
# ragbot.eval.{step_name}.aggregate_score
```

---

## Breaking Changes

⚠️ **Removed:**
- `base.py` - Custom evaluator base classes
  - ❌ `BaseEvaluator`
  - ❌ `DeterministicEvaluator`
  - ❌ `LLMEvaluator`
  - ❌ `EvaluationScore` (replaced by Phoenix Score wrapper)

- Class-based evaluators:
  - ❌ `CorrectnessEvaluator` class
  - ❌ `RelevanceEvaluator` class
  - ❌ `GroundednessEvaluator` class
  - ❌ `RoutingEvaluator` class
  - ❌ `SafetyEvaluator` class
  - ❌ `LatencyEvaluator` class
  - ❌ `FormatEvaluator` class

✅ **Replacement:** Import factory functions instead
```python
# ❌ Old
from ragbot.evaluations import CorrectnessEvaluator
evaluator = CorrectnessEvaluator()

# ✅ New
from ragbot.evaluations.correctness import create_correctness_evaluator
evaluator = create_correctness_evaluator()
```

---

## Migration Checklist

- [ ] Update imports from class-based to factory functions
- [ ] Set `GOOGLE_API_KEY` or relevant LLM provider credentials
- [ ] Update test code to use new factory functions
- [ ] Verify `EvaluationRunner` batch operations still work
- [ ] Test offline evaluation pipeline
- [ ] Validate chat graph integration (should work as-is)
- [ ] Run full test suite with credentials configured
- [ ] Update any custom evaluator subclasses to factory pattern

---

## Testing

### Factory Functions
```python
from ragbot.evaluations.correctness import create_correctness_evaluator

evaluator = create_correctness_evaluator()
assert evaluator is not None
```

### Runner
```python
from ragbot.evaluations import EvaluationRunner

runner = EvaluationRunner()
result = await runner.evaluate(
    input_text="Test?",
    response_text="Answer."
)
assert result.aggregate_score() >= 0.0
```

### Offline Pipeline
```pytest
pytest tests/eval/test_runner.py -v
pytest tests/eval/test_offline.py -v
pytest tests/test_evaluation_integration.py -v
```

**Note:** Tests requiring LLM calls need `GOOGLE_API_KEY` configured.

---

## Benefits of Phoenix SDK Migration

| Benefit | Details |
|---------|---------|
| **LLM-as-Judge** | Use actual LLMs for correctness/relevance (not heuristics) |
| **Built-in Evaluators** | Use proven Phoenix evaluators (FaithfulnessEvaluator, etc.) |
| **Batch Efficiency** | Phoenix's `evaluate_dataframe()` optimizes parallel evaluation |
| **Standardization** | Consistent API across all evaluator types |
| **Maintenance** | Rely on Phoenix's maintenance, updates, best practices |
| **Scalability** | Better support for large-scale evaluation runs |
| **Integration** | Seamless Phoenix dashboard/monitoring integration |

---

## Support

For Phoenix SDK questions, see:
- [Phoenix Documentation](https://docs.runphoenix.com)
- [Phoenix GitHub](https://github.com/arize-ai/phoenix)
- [Evaluation Examples](https://docs.runphoenix.com/evaluation)
