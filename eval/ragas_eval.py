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
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from backend.app.chat.pipeline import run_chat
from backend.app.config import PROJECT_ROOT, settings
from backend.app.models.schemas import ChatRequest, ResponseType
from backend.app.observability import flush as flush_traces, propagate_attributes
from backend.app.rag.retriever import RetrievedChunk
from backend.app.sources import priority_order
from eval.run_eval import EVAL_DIR, PROFILES_PATH, CASES_PATH, RESULTS_DIR
from eval.schemas import EvalSuite, TestCase

# RAGAS and LangChain read OPENAI_API_KEY from os.environ, not from our
# pydantic-settings object. Mirror the key across BEFORE importing ragas /
# langchain — those modules cache the env at import time. If the key is
# missing entirely, the import-time check below produces a clearer error
# than the deep stack trace from a downstream OpenAI client.
if not settings.openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set in .env — required for the RAGAS judge and embeddings."
    )
os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

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
    # retrieved_contexts is the list[str] RAGAS expects (text only).
    # chunks carries the full metadata (source, score) for reporting.
    # Both are derived from the same pipeline capture — keep them in sync.
    retrieved_contexts: list[str]
    chunks: list[RetrievedChunk]
    trace_id: str | None
    elapsed_s: float


def run_and_capture(
    case: TestCase,
    suite: EvalSuite,
    eval_session_id: str,
) -> AnswerCase | None:
    """
    Run one test case. Return an AnswerCase only if the pipeline produced an
    answer (with chunks); return None for emergency / out_of_scope / no_results
    cases since RAGAS would score canned text and yield meaningless numbers.

    eval_session_id groups all traces from one eval run into a single Langfuse
    Session so you can see every case + its RAGAS scores in one view without
    clicking into individual traces.
    """
    profile = suite.profiles[case.profile]
    request = ChatRequest(message=case.query, user_profile=profile)
    capture: dict = {}

    # Each trace gets:
    #   trace_name  — the test case id ("amla_pregnancy") for direct search
    #   session_id  — the eval run timestamp ("ragas_eval_20260501_143022") so
    #                 every case from this run is grouped in one Session view
    #   tags        — ["ragas_eval", <category>] for cross-run filtering
    # propagate_attributes is a no-op when Langfuse is disabled.
    t0 = time.perf_counter()
    with propagate_attributes(
        trace_name=case.id,
        session_id=eval_session_id,
        tags=["ragas_eval", case.category.value],
    ):
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
        chunks=chunks,
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
    # Explicit api_key — LangChain's ChatOpenAI reads from os.environ
    # OPENAI_API_KEY by default, but pydantic-settings loaded our key into
    # `settings.openai_api_key`, not into the environment. Passing it through
    # avoids the user having to also export OPENAI_API_KEY just for eval.
    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=judge_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
    )
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

def attach_scores_to_langfuse(
    answer_cases: list[AnswerCase],
    scores_df,
    judge_model: str,
) -> int:
    """
    Loop the per-row RAGAS scores back to the Langfuse traces that produced
    each answer. Returns the count of successfully attached scores.

    Each score comment includes the case id, category, and judge model so
    that the Langfuse Scores table is self-explanatory — you can read off
    which case a score belongs to and what drove it without clicking into
    the trace.
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
                data_type="NUMERIC",
                comment=(
                    f"case={ac.case.id} "
                    f"category={ac.case.category.value} "
                    f"judge={judge_model}"
                ),
            )
            attached += 1

    langfuse.flush()
    return attached


# ---------- Console + markdown reporting ----------

def _format_source_breakdown(chunks: list[RetrievedChunk]) -> str:
    """
    Compact per-case source/score string for the markdown report table.

    Sources appear in project priority order (MoHFW → FOGSI → WHO).
    Within each source, scores are shown highest-first.

    Example: "MoHFW×2(0.91,0.88) FOGSI×1(0.74) WHO×1(0.71)"
    Returns "—" for an empty chunk list.
    """
    if not chunks:
        return "—"

    by_source: dict[str, list[float]] = defaultdict(list)
    for c in chunks:
        by_source[c.org_display_name].append(c.score)

    parts = []
    # Iterate in canonical priority order so the string is stable across runs.
    for source in priority_order():
        if source not in by_source:
            continue
        scores = sorted(by_source[source], reverse=True)
        parts.append(f"{source}×{len(scores)}({','.join(f'{s:.2f}' for s in scores)})")

    return " ".join(parts)


def _source_diversity_stats(answer_cases: list[AnswerCase]) -> dict:
    """
    Aggregate source diversity metrics across all scored cases.

    Returns a dict with:
      avg_unique_sources  — mean number of distinct sources per case
      source_presence     — {source: (count, pct)} cases containing ≥1 chunk
      source_avg_score    — {source: mean reranker score} across all chunks
      all_sources_cases   — count of cases where every known source contributed
    """
    n = len(answer_cases)
    sources = priority_order()

    presence: dict[str, int] = defaultdict(int)
    all_scores: dict[str, list[float]] = defaultdict(list)
    unique_counts: list[int] = []
    all_sources_cases = 0

    for ac in answer_cases:
        present = {c.org_display_name for c in ac.chunks}
        unique_counts.append(len(present))
        for src in present:
            presence[src] += 1
        for c in ac.chunks:
            all_scores[c.org_display_name].append(c.score)
        if present >= set(sources):
            all_sources_cases += 1

    return {
        "n": n,
        "avg_unique_sources": sum(unique_counts) / n if n else 0.0,
        "all_sources_cases": all_sources_cases,
        "source_presence": {
            s: (presence[s], presence[s] / n * 100) for s in sources
        },
        "source_avg_score": {
            s: (sum(all_scores[s]) / len(all_scores[s])) if all_scores[s] else None
            for s in sources
        },
    }


def _print_aggregate(scores_df, n_cases: int, answer_cases: list[AnswerCase]) -> None:
    """Print mean RAGAS scores and source diversity summary to stdout.

    Aggregates are computed from `to_pandas()` rather than EvaluationResult
    directly — RAGAS's EvaluationResult.__getitem__ expects int, not string,
    so `metric in result` raises KeyError(0).
    """
    print()
    print("=" * 72)
    print("  RAGAS AGGREGATE SCORES")
    print("=" * 72)
    print(f"  Cases scored: {n_cases}")
    for metric in METRIC_COLUMNS:
        if metric in scores_df.columns:
            mean = scores_df[metric].mean()
            print(f"  {metric:<45} {mean:.3f}")

    stats = _source_diversity_stats(answer_cases)
    print()
    print(f"  Avg unique sources per case: {stats['avg_unique_sources']:.1f}")
    print(f"  Cases with all sources:      {stats['all_sources_cases']}/{n_cases}")
    for src in priority_order():
        count, pct = stats["source_presence"].get(src, (0, 0.0))
        avg = stats["source_avg_score"].get(src)
        avg_str = f"avg score {avg:.3f}" if avg is not None else "no chunks"
        print(f"  {src:<10} present in {count:>2}/{n_cases} cases ({pct:4.0f}%)  {avg_str}")


def _markdown_report(
    answer_cases: list[AnswerCase],
    scores_df,
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

    # Compute aggregate from the DataFrame for the same reason as _print_aggregate.
    lines += ["## Aggregate scores", "", "| Metric | Score |", "|---|---|"]
    for metric in METRIC_COLUMNS:
        if metric in scores_df.columns:
            lines.append(f"| {metric} | {scores_df[metric].mean():.3f} |")
    lines.append("")

    lines += [
        "## Per-case scores",
        "",
        "| ID | Category | Faithfulness | Answer relevancy | Context precision | Retrieved |",
        "|---|---|---|---|---|---|",
    ]
    for ac, (_, row) in zip(answer_cases, scores_df.iterrows()):
        lines.append(
            f"| {ac.case.id} | {ac.case.category.value} "
            f"| {row.get('faithfulness', float('nan')):.3f} "
            f"| {row.get('answer_relevancy', float('nan')):.3f} "
            f"| {row.get('llm_context_precision_without_reference', float('nan')):.3f} "
            f"| {_format_source_breakdown(ac.chunks)} |"
        )
    lines.append("")

    # Source diversity aggregate — the primary lens for evaluating whether
    # retrieval is monopolised by one source and whether top_k is right.
    stats = _source_diversity_stats(answer_cases)
    n = stats["n"]
    lines += [
        "## Source diversity",
        "",
        f"**Avg unique sources per case:** {stats['avg_unique_sources']:.1f}  ",
        f"**Cases with all {len(priority_order())} sources:** "
        f"{stats['all_sources_cases']}/{n} "
        f"({stats['all_sources_cases']/n*100:.0f}%)",
        "",
        "| Source | Cases present | % of cases | Avg reranker score |",
        "|---|---|---|---|",
    ]
    for src in priority_order():
        count, pct = stats["source_presence"].get(src, (0, 0.0))
        avg = stats["source_avg_score"].get(src)
        avg_str = f"{avg:.3f}" if avg is not None else "—"
        lines.append(f"| {src} | {count}/{n} | {pct:.0f}% | {avg_str} |")
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

    # All traces from this run share one session_id so they appear together in
    # the Langfuse Sessions view. Format: "ragas_eval_<timestamp>" is both
    # human-readable and sortable. Printed here so you can find it in the UI.
    eval_session_id = f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Running pipeline on {len(cases)} case(s) to capture answers + chunks...")
    print(f"Langfuse session: {eval_session_id}")
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
                ac = run_and_capture(case, suite, eval_session_id)
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
    _print_aggregate(df, n_cases=len(answer_cases), answer_cases=answer_cases)

    # Send scores back to the Langfuse traces created during this run.
    if not args.no_langfuse_scores and settings.langfuse_enabled:
        n = attach_scores_to_langfuse(answer_cases, df, judge_model=args.judge_model)
        print(f"\n  Attached {n} score(s) to Langfuse traces.")
        print(f"  View session: https://cloud.langfuse.com/sessions/{eval_session_id}")
    elif not settings.langfuse_enabled:
        print("\n  (Langfuse not configured — skipping score attachment.)")

    if not args.no_report:
        path = write_markdown_report(
            answer_cases=answer_cases,
            scores_df=df,
            judge_model=args.judge_model,
            note=args.note,
            skipped=skipped,
        )
        print(f"\n  Report written: {path.relative_to(PROJECT_ROOT)}")

    flush_traces()
    return 0


if __name__ == "__main__":
    sys.exit(main())
