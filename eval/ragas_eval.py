"""
ragas_eval.py — Answer-quality evaluation layer using RAGAS.

Sits on top of the behavioural eval (eval/run_eval.py). Where run_eval.py asks
"did the pipeline route correctly?", this asks "given that an answer was
produced, was it any good?". The two layers are complementary, not redundant —
see the project's MEMORY.md for the layering rule.

What it does:
  1. Loads the same eval suite (user_profiles.yaml + test_cases.yaml).
  2. For each test case where the pipeline produces response_type=answer,
     runs the pipeline once and captures (user_input, response, retrieved
     chunks, Langfuse trace_id) via run_chat's _eval_capture hook.
  3. Builds a RAGAS EvaluationDataset from the captured tuples.
  4. Runs evaluate() with the three reference-free metrics — faithfulness,
     answer_relevancy, llm_context_precision_without_reference — using a
     stronger judge LLM than the answer LLM (per the ragas skill's rule
     against same-model judging).
  5. Attaches the per-case scores back to the original Langfuse traces via
     langfuse.create_score so they appear in the trace UI alongside the
     prompts and chunks.
  6. Writes a markdown report alongside the existing run_eval reports for
     diff-friendly tracking across tuning runs.

Usage:
    python -m eval.ragas_eval
    python -m eval.ragas_eval --category core_nutrition
    python -m eval.ragas_eval --case iron_basic
    python -m eval.ragas_eval --judge-model gpt-4o-mini
    python -m eval.ragas_eval -m "raised threshold to 0.55"
    python -m eval.ragas_eval --no-langfuse-scores  # local-only, no score attachment

Cost note: with the default 3 metrics × ~26 cases, this is ~250-500 judge LLM
calls per run. Budget accordingly. See .claude/skills/ragas/references/pitfalls.md.
"""

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

from backend.app.chat.pipeline import run_chat
from backend.app.config import PROJECT_ROOT, settings
from backend.app.models.schemas import ChatRequest, ResponseType
from backend.app.observability import flush as flush_traces
from eval.run_eval import EVAL_DIR, PROFILES_PATH, CASES_PATH, RESULTS_DIR
from eval.schemas import EvalSuite, TestCase

# Default judge: stronger than the answer LLM (gpt-4.1-nano) to avoid same-model
# bias. Override with --judge-model. Always temperature=0 for reproducibility.
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

# Metric column names that RAGAS uses in result.to_pandas() — these must match
# the names we send to Langfuse.create_score so dashboards stay searchable.
METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_without_reference",
)


# ---------- Loading (mirrors run_eval.py) ----------

def load_suite() -> EvalSuite:
    with open(PROFILES_PATH) as f:
        profiles_raw = yaml.safe_load(f)
    with open(CASES_PATH) as f:
        cases_raw = yaml.safe_load(f)
    return EvalSuite(profiles=profiles_raw, cases=cases_raw)


# ---------- Per-case execution with chunk + trace_id capture ----------

@dataclass
class AnswerCase:
    """One answer-producing case captured for RAGAS scoring."""
    case: TestCase
    user_input: str
    response: str
    retrieved_contexts: list[str]
    trace_id: str | None
    elapsed_s: float


def run_and_capture(case: TestCase, suite: EvalSuite) -> AnswerCase | None:
    """
    Run one test case. Return an AnswerCase only if the pipeline produced an
    answer (with chunks); return None for emergency / out_of_scope / no_results
    cases since RAGAS would score canned text and yield meaningless numbers.
    """
    profile = suite.profiles[case.profile]
    request = ChatRequest(message=case.query, user_profile=profile)
    capture: dict = {}

    t0 = time.perf_counter()
    response = run_chat(request, _eval_capture=capture)
    elapsed = time.perf_counter() - t0

    # Filter: only score real answers. Canned guardrail text would inflate
    # faithfulness (no claims to verify) and tank context_precision (no chunks).
    if response.response_type is not ResponseType.answer:
        return None

    chunks = capture.get("chunks") or []
    if not chunks:
        return None

    return AnswerCase(
        case=case,
        user_input=case.query,
        response=response.answer,
        retrieved_contexts=[c.text for c in chunks],
        trace_id=capture.get("trace_id"),
        elapsed_s=elapsed,
    )


# ---------- RAGAS evaluation ----------

def build_ragas_dataset(answer_cases: list[AnswerCase]) -> EvaluationDataset:
    """Map our AnswerCase shape onto RAGAS's required field names."""
    return EvaluationDataset.from_list([
        {
            "user_input": ac.user_input,
            "response": ac.response,
            "retrieved_contexts": ac.retrieved_contexts,
        }
        for ac in answer_cases
    ])


def score_with_ragas(answer_cases: list[AnswerCase], judge_model: str):
    """Run the three reference-free RAGAS metrics. Returns the result object."""
    dataset = build_ragas_dataset(answer_cases)
    judge = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    return evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ],
        llm=judge,
    )


# ---------- Langfuse score attachment ----------

def attach_scores_to_langfuse(answer_cases: list[AnswerCase], scores_df) -> int:
    """
    Loop the per-row RAGAS scores back to the Langfuse traces that produced
    each answer. Returns the count of successfully attached scores.
    """
    if not settings.langfuse_enabled:
        return 0

    from langfuse import get_client
    langfuse = get_client()
    attached = 0

    for ac, (_, row) in zip(answer_cases, scores_df.iterrows()):
        if ac.trace_id is None:
            continue
        for metric in METRIC_COLUMNS:
            if metric not in row:
                continue
            value = row[metric]
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            langfuse.create_score(
                trace_id=ac.trace_id,
                name=metric,
                value=value,
                comment=f"RAGAS {metric} via gpt-4o-mini judge",
            )
            attached += 1

    langfuse.flush()
    return attached


# ---------- Console + markdown reporting ----------

def _print_aggregate(result, n_cases: int) -> None:
    print()
    print("=" * 72)
    print("  RAGAS AGGREGATE SCORES")
    print("=" * 72)
    print(f"  Cases scored: {n_cases}")
    for metric in METRIC_COLUMNS:
        if metric in result:
            print(f"  {metric:<45} {result[metric]:.3f}")


def _markdown_report(
    answer_cases: list[AnswerCase],
    scores_df,
    aggregate,
    judge_model: str,
    note: str | None,
    timestamp: str,
    skipped: list[TestCase],
) -> str:
    lines: list[str] = [
        f"# Poshan Saathi — RAGAS Eval {timestamp}",
        "",
        f"**Cases scored:** {len(answer_cases)}",
        f"**Cases skipped (non-answer):** {len(skipped)}",
        f"**Judge model:** `{judge_model}` (temperature 0)",
        "",
    ]

    if note:
        lines += ["## Note", "", f"> {note}", ""]

    lines += ["## Aggregate scores", "", "| Metric | Score |", "|---|---|"]
    for metric in METRIC_COLUMNS:
        if metric in aggregate:
            lines.append(f"| {metric} | {aggregate[metric]:.3f} |")
    lines.append("")

    lines += [
        "## Per-case scores",
        "",
        "| ID | Category | Faithfulness | Answer relevancy | Context precision |",
        "|---|---|---|---|---|",
    ]
    for ac, (_, row) in zip(answer_cases, scores_df.iterrows()):
        lines.append(
            f"| {ac.case.id} | {ac.case.category.value} "
            f"| {row.get('faithfulness', float('nan')):.3f} "
            f"| {row.get('answer_relevancy', float('nan')):.3f} "
            f"| {row.get('llm_context_precision_without_reference', float('nan')):.3f} |"
        )
    lines.append("")

    if skipped:
        lines += [
            "## Skipped (non-answer cases)",
            "",
            "These produced emergency / out_of_scope / no_results responses and ",
            "were not scored — RAGAS metrics are meaningless on canned text.",
            "",
        ]
        for c in skipped:
            lines.append(f"- `{c.id}` ({c.category.value})")
        lines.append("")

    return "\n".join(lines)


def write_markdown_report(*args, **kwargs) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"ragas_{timestamp}.md"
    content = _markdown_report(*args, timestamp=timestamp, **kwargs)
    path.write_text(content)
    return path


# ---------- Filtering ----------

def _filter_cases(cases, category: str | None, case_id: str | None):
    if category:
        cases = [c for c in cases if c.category.value == category]
    if case_id:
        cases = [c for c in cases if c.id == case_id]
    return cases


# ---------- Entry point ----------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score RAG answers with RAGAS (reference-free metrics)."
    )
    parser.add_argument("--category", help="Run only cases in this category")
    parser.add_argument("--case", help="Run only the case with this id")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenAI model used as the judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--no-langfuse-scores",
        action="store_true",
        help="Skip attaching RAGAS scores to Langfuse traces",
    )
    parser.add_argument("--no-report", action="store_true", help="Skip writing the markdown report")
    parser.add_argument(
        "-m", "--note",
        help="Short message about what changed since the last run, embedded in the report",
    )
    args = parser.parse_args()

    try:
        suite = load_suite()
    except ValidationError as e:
        print("Suite failed validation:\n")
        print(e)
        return 2

    cases = _filter_cases(suite.cases, args.category, args.case)
    if not cases:
        print("No cases match the given filters.")
        return 1

    print(f"Running pipeline on {len(cases)} case(s) to capture answers + chunks...")
    if args.note:
        print(f"Note: {args.note}")
    print()

    answer_cases: list[AnswerCase] = []
    skipped: list[TestCase] = []
    by_category: dict[str, list[TestCase]] = defaultdict(list)
    for c in cases:
        by_category[c.category.value].append(c)

    for category in sorted(by_category):
        print(f"[{category}]")
        for case in by_category[category]:
            try:
                ac = run_and_capture(case, suite)
            except Exception as e:
                print(f"  ✗ {case.id:<28} pipeline raised: {type(e).__name__}: {e}")
                continue
            if ac is None:
                print(f"  - {case.id:<28} skipped (non-answer)")
                skipped.append(case)
            else:
                print(f"  ✓ {case.id:<28} captured  ({ac.elapsed_s:.2f}s)")
                answer_cases.append(ac)
        print()

    if not answer_cases:
        print("No answer-producing cases — nothing for RAGAS to score.")
        flush_traces()
        return 1

    print(f"Scoring {len(answer_cases)} case(s) with RAGAS judge={args.judge_model}...")
    print("(Each case ≈ 3-9 judge LLM calls. This takes a minute.)")
    print()

    result = score_with_ragas(answer_cases, judge_model=args.judge_model)
    df = result.to_pandas()
    _print_aggregate(result, n_cases=len(answer_cases))

    # Send scores back to the Langfuse traces created during this run.
    if not args.no_langfuse_scores and settings.langfuse_enabled:
        n = attach_scores_to_langfuse(answer_cases, df)
        print(f"\n  Attached {n} score(s) to Langfuse traces.")
    elif not settings.langfuse_enabled:
        print("\n  (Langfuse not configured — skipping score attachment.)")

    if not args.no_report:
        path = write_markdown_report(
            answer_cases=answer_cases,
            scores_df=df,
            aggregate=result,
            judge_model=args.judge_model,
            note=args.note,
            skipped=skipped,
        )
        print(f"\n  Report written: {path.relative_to(PROJECT_ROOT)}")

    flush_traces()
    return 0


if __name__ == "__main__":
    sys.exit(main())
