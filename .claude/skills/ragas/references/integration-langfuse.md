# Integrating RAGAS with Langfuse

RAGAS produces per-row scores; Langfuse holds the corresponding RAG traces. Wiring them together gives you a UI where each trace shows its faithfulness, relevancy, and context-precision side-by-side with the prompt, retrieved chunks, and answer — without leaving the trace view.

## The pattern

There are two common architectures. Pick one based on whether RAGAS runs *during* the trace or *after* it.

### Pattern A: post-hoc score attachment (recommended)

Run RAGAS as a separate evaluation pass over traces that already exist in Langfuse. This is the canonical pattern for nightly evals, CI gates, and one-off analysis.

```python
from langfuse import get_client
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI

# 1. Pull the inputs you need from your traces. The exact extraction depends on
#    your pipeline — typically you store user_input, response, and retrieved
#    contexts as span input/output during the trace.
trace_records = [
    # {"trace_id": "...", "case_id": "...", "category": "...", "user_input": "...", "response": "...", "retrieved_contexts": [...]}
]

# 2. Build the RAGAS dataset (note the field-name renaming).
dataset = EvaluationDataset.from_list([
    {
        "user_input": r["user_input"],
        "response": r["response"],
        "retrieved_contexts": r["retrieved_contexts"],
    }
    for r in trace_records
])

# 3. Run the evaluation.
judge_model = "gpt-4o-mini"
judge = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()],
    llm=judge,
)
df = result.to_pandas()  # one row per sample, one column per metric

# 4. Loop the per-row scores back to the originating Langfuse traces.
#    Include case_id and category in the comment so the Scores table is
#    self-explanatory — you can read which case a score belongs to without
#    clicking into the trace.
langfuse = get_client()
metric_columns = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference"]

for trace_record, (_, row) in zip(trace_records, df.iterrows()):
    for metric in metric_columns:
        if metric in row and row[metric] is not None:
            langfuse.create_score(
                trace_id=trace_record["trace_id"],
                name=metric,
                value=float(row[metric]),
                data_type="NUMERIC",
                comment=(
                    f"case={trace_record['case_id']} "
                    f"category={trace_record['category']} "
                    f"judge={judge_model}"
                ),
            )
langfuse.flush()
```

After this runs, every trace in Langfuse has the three scores attached. You can filter "show me traces where faithfulness < 0.7", build dashboards, or compare runs.

### Pattern B: in-line scoring per request (rarely correct)

Calling RAGAS during a live `/chat` request adds 5-15 seconds of judge-LLM latency per user request, and triples cost. **Don't do this** unless you have a specific real-time monitoring need that justifies it. Pattern A scales better, can be batched, and runs on a schedule rather than a critical path.

## Making scores navigable by test case in the Langfuse UI

Attaching scores to traces is only half the job. The other half is structuring traces so you can compare cases across an eval run without clicking into each one individually. Use all three mechanisms together:

### 1. Session grouping (most important)

Generate a single `session_id` at the start of the eval run and pass it to every trace via `propagate_attributes`. All traces then appear under one Session in the Langfuse Sessions view, where you can see each case ID as a row with its scores side-by-side.

```python
from langfuse import propagate_attributes
from datetime import datetime

eval_session_id = f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with propagate_attributes(
    trace_name=case_id,          # e.g. "amla_pregnancy" — makes each trace findable
    session_id=eval_session_id,  # groups all cases from this run together
    tags=["ragas_eval", category],  # enables filtering by eval type and category
):
    response = run_pipeline(...)
```

**In the Langfuse UI**: go to Sessions → search for `ragas_eval_<timestamp>`. You'll see all cases from the run listed with their latency, scores, and tags — no manual clicking required.

### 2. Trace naming by case ID

`trace_name=case_id` (e.g. `"amla_pregnancy"`) means you can search for any specific case directly in the Traces view. Without this, all traces appear as `"chat"` and are indistinguishable.

### 3. Tags for cross-run filtering

`tags=["ragas_eval", "core_nutrition"]` lets you filter the Traces view to show only eval runs, or only runs for a particular category, across all sessions. Useful when comparing category-level scores between two tuning runs.

### What the UI looks like after wiring

| View | What you see |
|---|---|
| Sessions → `ragas_eval_*` | All cases in the run, each row = one trace, scores visible inline |
| Traces → filter by tag `ragas_eval` | Every eval trace ever run, sortable by score |
| Trace detail → Scores tab | `faithfulness`, `answer_relevancy`, `llm_context_precision_without_reference` for that case |
| Traces → filter score `faithfulness < 0.5` | Cases where the answer wasn't grounded — triage list |

## Field-name mapping

RAGAS and your typical app/Langfuse schemas use different vocabulary for the same concepts. Map them explicitly when building the dataset:

| Your app / Langfuse trace | RAGAS field          |
| ------------------------- | -------------------- |
| `query` or `message`      | `user_input`         |
| `answer` or `output`      | `response`           |
| `chunks[].text` (list)    | `retrieved_contexts` |
| `expected_answer` (if any)| `reference`          |

`retrieved_contexts` must be a `list[str]`. If your trace stores chunks as a list of objects (with `.text`, `.page`, etc.), extract just the text.

## What metric names show up in Langfuse

When RAGAS scores arrive in Langfuse via `create_score(name=...)`, the names you pass become the score keys you filter by in the UI. Use the canonical RAGAS column names so they match the docs and stay searchable:

- `faithfulness`
- `answer_relevancy` (column name even though the class is `ResponseRelevancy`)
- `llm_context_precision_without_reference`
- `llm_context_recall`
- `factual_correctness` (if using GT)

Don't rename them on the way in — consistency matters more than brevity.

## Alternative: dataset-based evaluation in Langfuse

Langfuse also supports running RAGAS against a **Langfuse Dataset** (a curated set of cases stored in Langfuse's UI). The flow is the same — pull the dataset, build the RAGAS evaluation set, run, write scores back. See `langfuse.com/docs/datasets` for the dataset API; the RAGAS half doesn't change.

## Cost note

Each RAGAS run replays the judge LLM over every (sample × metric). On a 100-trace nightly eval with 3 metrics this is ~300-600 judge calls + embedding calls. Budget the cost like any other batch job; see `pitfalls.md` for cost-control tactics.
