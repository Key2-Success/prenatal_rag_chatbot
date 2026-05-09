"""
ragas_eval.py — Combined routing + answer-quality evaluation.

Two complementary layers in one report:

  Layer 1 — Routing (all 26 cases):
    Did the pipeline send each message to the right destination?
    Asserts: response_type matches expected.behavior, and (for answer cases)
    first cited org matches expected.cites_org.

  Layer 2 — Answer quality (the ~17 answer-producing cases only):
    Given that an answer was produced, was it any good?
    Metrics: faithfulness, answer_relevancy, llm_context_precision_without_reference.
    RAGAS metrics are meaningless on canned emergency/guardrail text — those are
    always skipped from RAGAS scoring but still appear in the routing section.

The two layers are complementary, not redundant — see MEMORY.md for the
layering rule.

Usage:
    python -m eval.ragas_eval
    python -m eval.ragas_eval --category core_nutrition
    python -m eval.ragas_eval --case iron_basic
    python -m eval.ragas_eval --judge-model gpt-4o-mini
    python -m eval.ragas_eval -m "raised threshold to 0.55"
    python -m eval.ragas_eval --no-langfuse-scores  # local-only, no score attachment

Cost note: with the default 3 metrics × ~17 answer cases, this is ~150-350 judge LLM
calls per run. Budget accordingly.
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
# langchain — those modules cache the env at import time.
if not settings.openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set in .env — required for embeddings (ResponseRelevancy)."
    )
os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
# Mirror the Anthropic key the same way — LangChain's ChatAnthropic reads
# os.environ["ANTHROPIC_API_KEY"] by default, and pydantic-settings has loaded
# it into settings.anthropic_api_key. We don't fail loud here because the rest
# of the app doesn't need it; score_with_ragas() validates at use-time.
if settings.anthropic_api_key:
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402  (kept for opt-in fallback)
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402

# Default judge: cross-vendor (Anthropic Claude judging OpenAI gpt-4.1-nano
# answers). Two best practices from the RAGAS skill (references/pitfalls.md #2
# and references/judge-config.md):
#   1. Judge must be a different model from the answer LLM. Different VENDOR
#      is even better — a model judging output from its own family tends to
#      score generously and shares failure modes. (Pitfall #2 says verbatim:
#      "Different vendor is even better (e.g. answer LLM is OpenAI, judge is
#      Claude).")
#   2. Judge must be at least as strong as the answer LLM. Claude Sonnet >>
#      gpt-4.1-nano on every reasoning benchmark, so the detection ceiling is
#      well above the answer model — it can catch errors the answer model
#      makes.
# Pinned to a dated snapshot, not the floating "claude-sonnet-4-5" alias —
# eval scores need version stability for longitudinal comparison (RAGAS
# pitfalls.md #5).
#
# Note on reproducibility: temperature=0 reduces but does NOT eliminate
# variance. RAGAS metrics are multi-step LLM pipelines (claim extraction →
# verification, etc.) so single-token flips cascade. Expect run-to-run drift
# of ~0.05-0.10; only deltas larger than that are meaningful. Run 3× and
# average for canonical numbers.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# Concurrency cap for RAGAS's parallel judge calls. RAGAS defaults can fan out
# fast enough to hit per-minute token limits on small accounts (Tier 1 OpenAI
# is 30k TPM; Anthropic tiers similar). 4 workers is the conservative end of
# the skill's suggested 4-8 range — favours reliability over wall-clock speed.
RAGAS_MAX_WORKERS = 4

# Metric column names that RAGAS uses in result.to_pandas() — these must match
# the names we send to Langfuse.create_score so dashboards stay searchable.
METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_without_reference",
)

# Truncate answers when echoing into the report — full answers would make the
# report hundreds of lines and noisy to diff.
_PREVIEW_CHARS = 160


# ---------- Loading (mirrors run_eval.py) ----------

def load_suite() -> EvalSuite:
    with open(PROFILES_PATH) as f:
        profiles_raw = yaml.safe_load(f)
    with open(CASES_PATH) as f:
        cases_raw = yaml.safe_load(f)
    return EvalSuite(profiles=profiles_raw, cases=cases_raw)


# ---------- Result types ----------

@dataclass
class RoutingResult:
    """
    Routing correctness for one test case — populated for all 26 cases.

    Mirrors run_eval.py's CaseResult so the routing section of this report
    is identical in structure and can be compared directly.
    """
    case: TestCase
    passed: bool
    reason: str
    elapsed_s: float
    actual_type: ResponseType | None
    actual_org: str | None
    answer_preview: str | None  # first _PREVIEW_CHARS of the actual response


@dataclass
class AnswerCase:
    """
    One answer-producing case captured for RAGAS scoring — populated only
    when response_type=answer. Holds both the text RAGAS needs and the full
    chunk metadata for the source-diversity section of the report.
    """
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


# ---------- Per-case execution ----------

def run_and_capture(
    case: TestCase,
    suite: EvalSuite,
    run_id: str,
    run_idx: int = 0,
    total_runs: int = 1,
) -> tuple[RoutingResult, AnswerCase | None]:
    """
    Run one test case end-to-end.

    Always returns a RoutingResult (all 26 cases are routing-evaluated).
    Also returns an AnswerCase when the pipeline produced response_type=answer;
    returns None in that slot for emergency / out_of_scope / no_results because
    RAGAS metrics are meaningless on canned text.

    run_id groups all traces from one eval run into a single Langfuse
    Session so you can see every case + its RAGAS scores in one view.

    For multi-run evals (total_runs > 1), each trace is tagged
    "run_<i>_of_<n>" so you can filter the session view by run iteration.
    """
    profile = suite.profiles[case.profile]
    request = ChatRequest(message=case.query, user_profile=profile)
    capture: dict = {}

    tags = ["ragas_eval", case.category.value]
    if total_runs > 1:
        tags.append(f"run_{run_idx + 1}_of_{total_runs}")

    t0 = time.perf_counter()
    try:
        with propagate_attributes(
            trace_name=case.id,
            session_id=run_id,
            tags=tags,
        ):
            response = run_chat(request, _eval_capture=capture)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        routing = RoutingResult(
            case=case,
            passed=False,
            reason=f"pipeline raised: {type(e).__name__}: {e}",
            elapsed_s=elapsed,
            actual_type=None,
            actual_org=None,
            answer_preview=None,
        )
        return routing, None

    elapsed = time.perf_counter() - t0
    actual_type = response.response_type
    actual_org = response.sources[0].org_display_name if response.sources else None
    preview = response.answer[:_PREVIEW_CHARS].replace("\n", " ") if response.answer else None

    # --- Routing evaluation (all cases) ---
    failures: list[str] = []
    if actual_type is not case.expected.behavior:
        failures.append(
            f"behavior: expected {case.expected.behavior.value}, "
            f"got {actual_type.value}"
        )
    if (
        case.expected.cites_org
        and actual_type is ResponseType.answer
        and actual_org != case.expected.cites_org
    ):
        failures.append(
            f"cites_org: expected {case.expected.cites_org}, got {actual_org}"
        )

    routing = RoutingResult(
        case=case,
        passed=not failures,
        reason="; ".join(failures) if failures else "ok",
        elapsed_s=elapsed,
        actual_type=actual_type,
        actual_org=actual_org,
        answer_preview=preview,
    )

    # --- RAGAS capture (answer cases only) ---
    chunks = capture.get("chunks") or []
    if actual_type is not ResponseType.answer or not chunks:
        return routing, None

    answer_case = AnswerCase(
        case=case,
        user_input=case.query,
        response=response.answer,
        retrieved_contexts=[c.text for c in chunks],
        chunks=chunks,
        trace_id=capture.get("trace_id"),
        elapsed_s=elapsed,
    )
    return routing, answer_case


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


def _build_judge(judge_model: str) -> LangchainLLMWrapper:
    """
    Construct the RAGAS judge LLM wrapper for the given model id.

    Selects the LangChain wrapper class by prefix:
      - "claude-..."  → ChatAnthropic (cross-vendor judge — the default)
      - "gpt-..."     → ChatOpenAI    (same-vendor; explicit opt-in only,
                                       e.g. when an Anthropic key is
                                       unavailable)

    Fails loud at construction time if the required API key is missing —
    better to error before the (slow, expensive) RAGAS evaluation kicks off
    than to discover it three minutes in.
    """
    if judge_model.startswith("claude"):
        if not settings.anthropic_api_key:
            raise RuntimeError(
                f"Judge model '{judge_model}' requires ANTHROPIC_API_KEY in .env. "
                f"Either set the key or pass --judge-model gpt-4o (note: "
                f"same-vendor judging is discouraged — see RAGAS pitfalls #2)."
            )
        chat = ChatAnthropic(
            model=judge_model,
            temperature=0,
            api_key=settings.anthropic_api_key,
        )
    elif judge_model.startswith("gpt"):
        chat = ChatOpenAI(
            model=judge_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
    else:
        raise ValueError(
            f"Unknown judge model family: '{judge_model}'. "
            f"Expected a model id starting with 'claude-' or 'gpt-'."
        )
    return LangchainLLMWrapper(chat)


def score_with_ragas(answer_cases: list[AnswerCase], judge_model: str):
    """
    Run the three reference-free RAGAS metrics. Returns the result object.

    `RunConfig(max_workers=...)` caps RAGAS's parallel judge calls so we don't
    blow through per-minute token limits — a fan-out of 10+ workers will trip
    Tier-1 OpenAI/Anthropic accounts in seconds. See RAGAS judge-config.md
    "Concurrency / batch size".
    """
    dataset = build_ragas_dataset(answer_cases)
    judge = _build_judge(judge_model)
    return evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ],
        llm=judge,
        run_config=RunConfig(max_workers=RAGAS_MAX_WORKERS),
    )


# ---------- Multi-run score aggregation ----------

def _aggregate_score_dfs(dfs: list):
    """
    Aggregate per-run RAGAS score DataFrames into one mean DataFrame plus a
    matching stddev DataFrame.

    Why average at all: RAGAS faithfulness and answer_relevancy are two-stage
    LLM pipelines (claim extraction → verification, or noncommittal-flag →
    synthetic-question generation). Both stages are stochastic — at temp=0
    the OpenAI/Anthropic APIs still drift across calls — so single-shot
    scores swing 0.05–0.30 run-to-run on the same answer. Averaging across
    N runs gives a value tight enough to compare against thresholds
    (e.g. "is faithfulness ≥ 0.9?") without the swing dominating.

    Inputs must all have the same row order — RAGAS preserves the order it
    received the dataset in, and we feed the same answer_cases list each
    run, so this holds. Non-metric columns (user_input, response, ...) are
    taken from dfs[0]; only METRIC_COLUMNS are averaged.

    For N=1 returns (dfs[0], None) so callers can branch on `stds is None`
    to skip the variance display.
    """
    if len(dfs) == 1:
        return dfs[0], None

    import numpy as np
    means = dfs[0].copy()
    stds = dfs[0].copy()
    for metric in METRIC_COLUMNS:
        if all(metric in df.columns for df in dfs):
            stacked = np.stack([df[metric].to_numpy() for df in dfs])
            means[metric] = stacked.mean(axis=0)
            stds[metric] = stacked.std(axis=0)
    return means, stds


def _fmt_score(mean: float, std: float | None) -> str:
    """Format a score as 'mean' (single run) or 'mean ± std' (multi-run)."""
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


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
    the Langfuse Scores table is self-explanatory.
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


# ---------- Source diversity helpers ----------

def _format_source_breakdown(chunks: list[RetrievedChunk]) -> str:
    """
    Compact per-case source/score string for the markdown report table.

    Sources appear in project priority order (MoHFW → FOGSI → WHO).
    Example: "MoHFW×2(0.91,0.88) FOGSI×1(0.74) WHO×1(0.71)"
    Returns "—" for an empty chunk list.
    """
    if not chunks:
        return "—"

    by_source: dict[str, list[float]] = defaultdict(list)
    for c in chunks:
        by_source[c.org_display_name].append(c.score)

    parts = []
    for source in priority_order():
        if source not in by_source:
            continue
        scores = sorted(by_source[source], reverse=True)
        parts.append(f"{source}×{len(scores)}({','.join(f'{s:.2f}' for s in scores)})")

    return " ".join(parts)


def _source_diversity_stats(answer_cases: list[AnswerCase]) -> dict:
    """Aggregate source diversity metrics across all scored cases."""
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


# ---------- Console output ----------

def _group_by_category(results: list[RoutingResult]) -> dict[str, list[RoutingResult]]:
    by_category: dict[str, list[RoutingResult]] = defaultdict(list)
    for r in results:
        by_category[r.case.category.value].append(r)
    return by_category


def _flaky_cases(all_routing_results: list[list[RoutingResult]]) -> dict[str, int]:
    """Map case_id → number of runs in which that case PASSED.

    Used to surface routing flakiness: a case that passes 2 of 3 runs is
    visibly different from one that passes all 3. Without this view a single
    flaky case can hide inside an "average pass rate" number.
    """
    pass_count: dict[str, int] = defaultdict(int)
    for run in all_routing_results:
        for r in run:
            if r.passed:
                pass_count[r.case.id] += 1
    return dict(pass_count)


def _print_routing_summary(
    all_routing_results: list[list[RoutingResult]],
    overall_elapsed_s: float,
) -> None:
    """
    Print the routing summary across N runs.

    For N==1 this collapses to the previous single-run view. For N>1 we show
    per-run pass counts (so a flaky case is visible as e.g. 25/26, 26/26,
    25/26 instead of being averaged into 25.3) plus the average and any
    cases that didn't pass every run.
    """
    n_runs = len(all_routing_results)
    print()
    print("=" * 72)
    suffix = f"  (across {n_runs} runs)" if n_runs > 1 else ""
    print(f"  ROUTING SUMMARY{suffix}")
    print("=" * 72)

    # Category-level pass counts come from run 1 (categorisation is fixed by
    # the test suite; the run dimension only affects pass/fail per case).
    by_category = _group_by_category(all_routing_results[0])
    for category in sorted(by_category):
        items = by_category[category]
        passed = sum(1 for r in items if r.passed)
        total = len(items)
        bar = "█" * passed + "░" * (total - passed)
        suffix = "  (run 1)" if n_runs > 1 else ""
        print(f"  {category:<25} {passed}/{total}  {bar}{suffix}")

    if n_runs > 1:
        print()
        print(f"  Per-run pass counts (out of {len(all_routing_results[0])}):")
        for run_idx, run in enumerate(all_routing_results):
            n_pass = sum(1 for r in run if r.passed)
            n_total = len(run)
            print(f"    run {run_idx + 1}: {n_pass}/{n_total}")

        avg_pass = sum(
            sum(1 for r in run if r.passed) for run in all_routing_results
        ) / n_runs
        n_total = len(all_routing_results[0])
        print(f"    average:  {avg_pass:.1f}/{n_total}")

        # Surface flaky cases — passed some runs, failed others.
        pass_count = _flaky_cases(all_routing_results)
        flaky = [
            (cid, count) for cid, count in pass_count.items()
            if 0 < count < n_runs
        ]
        # Cases that failed every run are NOT flaky — they're consistently failing.
        # List those separately.
        all_case_ids = {r.case.id for r in all_routing_results[0]}
        always_failing = sorted(all_case_ids - set(pass_count.keys()))

        if flaky:
            print()
            print(f"  Flaky cases (passed some runs, not others):")
            for cid, count in sorted(flaky):
                print(f"    {cid:<35} passed {count}/{n_runs} runs")
        if always_failing:
            print()
            print(f"  Consistently failing cases:")
            for cid in always_failing:
                print(f"    {cid}")

    print()
    print(f"  Wall-clock total: {overall_elapsed_s:.1f}s "
          f"(includes pipeline + RAGAS scoring across all {n_runs} run(s))")


def _print_ragas_aggregate(
    means_df,
    stds_df,
    n_cases: int,
    n_runs: int,
    answer_cases: list[AnswerCase],
) -> None:
    print()
    print("=" * 72)
    suffix = f" (averaged over {n_runs} runs)" if n_runs > 1 else ""
    print(f"  RAGAS AGGREGATE SCORES{suffix}")
    print("=" * 72)
    print(f"  Cases scored: {n_cases}")
    for metric in METRIC_COLUMNS:
        if metric in means_df.columns:
            mean = means_df[metric].mean()
            if stds_df is not None and metric in stds_df.columns:
                # Pool variance across cases AND runs to get an honest
                # uncertainty on the aggregate. The per-case std is run-to-run
                # noise; we report mean of those alongside the cross-case mean
                # so the reader sees the typical judge variance per case.
                avg_per_case_std = stds_df[metric].mean()
                print(f"  {metric:<45} {mean:.3f}  (±{avg_per_case_std:.3f} per-case judge variance)")
            else:
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


# ---------- Markdown report ----------

def _markdown_report(
    routing_results: list[RoutingResult],            # run-1 results, for the per-case detail table
    all_routing_results: list[list[RoutingResult]],  # all N runs, for aggregate pass-rate stats
    overall_elapsed_s: float,                         # wall-clock total across runs + scoring
    answer_cases: list[AnswerCase],
    scores_df,           # means_df when n_runs > 1
    stds_df,             # None when n_runs == 1
    n_runs: int,
    judge_model: str,
    note: str | None,
    timestamp: str,
) -> str:
    n_total = len(routing_results)
    n_scored = len(answer_cases)

    # Per-case pass count across runs — used both in the summary and in the
    # detail table to surface flakiness (case passes 2/3 runs, etc.).
    pass_count_by_id = _flaky_cases(all_routing_results)

    # Per-run pass counts.
    per_run_passed = [
        sum(1 for r in run if r.passed) for run in all_routing_results
    ]
    avg_passed = sum(per_run_passed) / n_runs
    avg_pass_rate = (avg_passed / n_total * 100) if n_total else 0.0

    runs_note = f"  **Runs:** {n_runs}" if n_runs > 1 else ""
    lines: list[str] = [
        f"# Poshan Saathi — Eval {timestamp}",
        "",
        f"**Wall-clock total:** {overall_elapsed_s:.1f}s "
        f"(pipeline + RAGAS scoring across all {n_runs} run{'s' if n_runs > 1 else ''})  "
        f"**Judge model:** `{judge_model}` (temperature 0){runs_note}",
        "",
    ]

    if note:
        lines += ["## Note", "", f"> {note}", ""]

    # ── Executive summary ────────────────────────────────────────────────────
    # Two layers, side-by-side. The reader should be able to answer "is this
    # run better or worse than the last one?" without scrolling past this block.
    lines += [
        "## Summary",
        "",
        "Two evaluation layers: **routing** (did the pipeline send the message "
        "to the right destination?) and **RAGAS answer-quality** (given that an "
        "answer was produced, was it any good?).",
        "",
        "### Layer 1 — Routing",
        "",
    ]

    if n_runs > 1:
        # Multi-run: show per-run pass counts AND the average. A single
        # collapsed number would hide a flaky case (passes 2 of 3 runs).
        lines += [
            f"**Average: {avg_passed:.1f}/{n_total} passed ({avg_pass_rate:.0f}%)** "
            f"across {n_runs} runs. Per-run breakdown:",
            "",
            "| Run | Passed | Total |",
            "|---|---|---|",
        ]
        for i, n_pass in enumerate(per_run_passed):
            lines.append(f"| {i + 1} | {n_pass} | {n_total} |")
        lines.append("")

        # Flaky cases — passed some runs, failed others. Distinct from
        # consistently-failing cases, which need different triage.
        all_case_ids = [r.case.id for r in routing_results]  # preserve order
        flaky = [
            (cid, pass_count_by_id.get(cid, 0))
            for cid in all_case_ids
            if 0 < pass_count_by_id.get(cid, 0) < n_runs
        ]
        always_failing = [
            cid for cid in all_case_ids if pass_count_by_id.get(cid, 0) == 0
        ]
        if flaky:
            lines += [
                "**Flaky cases** (non-deterministic routing across runs — should be 0):",
                "",
            ]
            for cid, count in flaky:
                lines.append(f"- `{cid}` — passed {count}/{n_runs} runs")
            lines.append("")
        if always_failing:
            lines += [
                "**Consistently failing cases**:",
                "",
            ]
            for cid in always_failing:
                lines.append(f"- `{cid}`")
            lines.append("")
    else:
        n_routing_passed = per_run_passed[0]
        routing_pass_rate = (n_routing_passed / n_total * 100) if n_total else 0.0
        lines += [
            f"**{n_routing_passed}/{n_total} passed ({routing_pass_rate:.0f}%)** "
            "across all cases. Checks `response_type` matches expected behavior, "
            "and (for answer cases) the first cited source matches `cites_org`.",
            "",
        ]

    lines += [
        "| Category | Passed | Total |",
        "|---|---|---|",
    ]
    by_category = _group_by_category(routing_results)
    suffix = "  _(run 1)_" if n_runs > 1 else ""
    for category in sorted(by_category):
        items = by_category[category]
        passed = sum(1 for r in items if r.passed)
        lines.append(f"| {category}{suffix} | {passed} | {len(items)} |")
    lines.append("")

    lines += [
        "### Layer 2 — RAGAS answer quality",
        "",
    ]
    if scores_df is None or n_scored == 0:
        lines += [
            "_No answer-producing cases this run — RAGAS scoring was skipped._",
            "",
        ]
    else:
        averaged = f" averaged over {n_runs} runs" if n_runs > 1 else ""
        lines += [
            f"**{n_scored} answer case(s)** scored on three reference-free "
            f"metrics{averaged}. Higher is better; 1.0 is the ceiling.",
            "",
        ]
        if n_runs > 1:
            lines += [
                "_The_ ± _figure is the average per-case stddev across runs — a"
                " direct measure of RAGAS judge variance for this run. Anything"
                " under ~0.05 is tight; over ~0.15 means the judge is unstable on"
                " these cases and the mean should be read with caution._",
                "",
                "| Metric | Mean ± per-case stddev | What it measures |",
                "|---|---|---|",
            ]
        else:
            lines += [
                "| Metric | Mean score | What it measures |",
                "|---|---|---|",
            ]
        metric_descriptions = {
            "faithfulness": "Are the answer's claims supported by the retrieved chunks?",
            "answer_relevancy": "Does the answer actually address the question asked?",
            "llm_context_precision_without_reference": "Are the retrieved chunks relevant to the question?",
        }
        for metric in METRIC_COLUMNS:
            if metric in scores_df.columns:
                mean = scores_df[metric].mean()
                if stds_df is not None and metric in stds_df.columns:
                    avg_std = stds_df[metric].mean()
                    score_str = f"{mean:.3f} ± {avg_std:.3f}"
                else:
                    score_str = f"{mean:.3f}"
                lines.append(
                    f"| {metric} | {score_str} | {metric_descriptions[metric]} |"
                )
        lines.append("")

    # ── Section 1: Routing breakdown ─────────────────────────────────────────
    multi_run_note = (
        f" The Pass column shows runs-passed-out-of-{n_runs}; "
        f"anything other than {n_runs}/{n_runs} is non-deterministic and "
        "needs investigation. Other columns (Actual, Cites, Time) are from run 1."
    ) if n_runs > 1 else ""
    pass_header = f"Pass ({n_runs} runs)" if n_runs > 1 else "Status"
    lines += [
        "## Routing breakdown",
        "",
        "Per-case detail for all cases — useful when a category in the summary "
        f"above is below 100% and you need to find the failing case quickly.{multi_run_note}",
        "",
        f"| ID | Category | {pass_header} | Expected | Actual | Cites (exp → got) | Time | Reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in routing_results:
        if n_runs > 1:
            n_passed = pass_count_by_id.get(r.case.id, 0)
            status = f"{n_passed}/{n_runs}"
        else:
            status = "PASS" if r.passed else "FAIL"
        actual = r.actual_type.value if r.actual_type else "—"
        cites = f"{r.case.expected.cites_org or '—'} → {r.actual_org or '—'}"
        lines.append(
            f"| {r.case.id} | {r.case.category.value} | {status} "
            f"| {r.case.expected.behavior.value} | {actual} "
            f"| {cites} | {r.elapsed_s:.2f}s | {r.reason} |"
        )
    lines.append("")

    # Answer previews for ALL cases (canned guardrail text included so you can
    # confirm at a glance that emergency / out_of_scope responses still read well).
    lines += ["### Answer previews (all cases)", ""]
    for r in routing_results:
        actual = r.actual_type.value if r.actual_type else "error"
        lines += [
            f"**{r.case.id}** — _{actual}_",
            "",
            f"> {r.answer_preview or '(no response)'}",
            "",
        ]

    # ── Section 2: RAGAS breakdown ───────────────────────────────────────────
    if scores_df is not None and answer_cases:
        lines += [
            "## RAGAS breakdown",
            "",
            "Per-case scores for every answer-producing case. The Retrieved "
            "column shows which sources contributed and their reranker scores, "
            "so a low context-precision score can be diagnosed against the "
            "actual chunks that drove it.",
            "",
            "| ID | Category | Faithfulness | Answer relevancy | Context precision | Retrieved |",
            "|---|---|---|---|---|---|",
        ]
        for i, (ac, (_, row)) in enumerate(zip(answer_cases, scores_df.iterrows())):
            std_row = stds_df.iloc[i] if stds_df is not None else None

            def _cell(metric: str) -> str:
                mean = row.get(metric, float("nan"))
                if std_row is None:
                    return f"{mean:.3f}"
                std = std_row.get(metric, float("nan"))
                return f"{mean:.3f} ± {std:.3f}"

            lines.append(
                f"| {ac.case.id} | {ac.case.category.value} "
                f"| {_cell('faithfulness')} "
                f"| {_cell('answer_relevancy')} "
                f"| {_cell('llm_context_precision_without_reference')} "
                f"| {_format_source_breakdown(ac.chunks)} |"
            )
        lines.append("")

    # ── Section 3: Source diversity ────────────────────────────────────────────
    if answer_cases:
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

    return "\n".join(lines)


def write_markdown_report(*args, run_id: str, **kwargs) -> Path:
    """Write the eval report to eval/results/<run_id>.md.

    `run_id` must be the same identifier used for the Langfuse session so that
    the filename and the Langfuse session URL share a single key — no more
    timestamp skew between what's on disk and what's in the trace dashboard.
    Format: "eval_YYYYMMDD_HHMMSS" (computed once in main(), passed here).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{run_id}.md"
    # Strip the "eval_" prefix so the markdown title reads "Eval 20260508_123456"
    # rather than "Eval eval_20260508_123456".
    timestamp = run_id.removeprefix("eval_")
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
        description="Run routing + RAGAS answer-quality evaluation."
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
        "--runs",
        type=int,
        default=1,
        help=(
            "Run the full pipeline + RAGAS scoring N times and average the "
            "scores. Recommended: 3 for canonical comparison numbers (RAGAS "
            "judge variance is the dominant source of run-to-run swing). "
            "Cost scales linearly: each extra run is another ~17 pipeline "
            "calls + ~150-300 judge calls. Default 1."
        ),
    )
    parser.add_argument(
        "-m", "--note",
        help="Short message about what changed since the last run, embedded in the report",
    )
    args = parser.parse_args()

    if args.runs < 1:
        print(f"--runs must be >= 1 (got {args.runs}).")
        return 2

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

    # Single identifier for both the Langfuse session and the markdown filename.
    # Computing it once here eliminates the timestamp skew that used to produce
    # different IDs in the trace dashboard vs. the saved report.
    # Format: "eval_YYYYMMDD_HHMMSS" — describes the artifact, not the tool.
    run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Running pipeline on {len(cases)} case(s)...")
    print(f"Run ID (Langfuse session + report filename): {run_id}")
    if args.note:
        print(f"Note: {args.note}")
    print()

    by_category: dict[str, list[TestCase]] = defaultdict(list)
    for c in cases:
        by_category[c.category.value].append(c)

    # First-run results are kept verbatim for display (routing table, answer
    # previews, source breakdown). Subsequent runs contribute (a) RAGAS score
    # DataFrames for variance-aware averaging, and (b) their full RoutingResult
    # list for routing-flakiness aggregation. Routing is nearly deterministic
    # (classifier at temp=0) but not guaranteed to be — surfacing per-run
    # pass counts and per-case pass-N-of-M counters catches the case where it
    # isn't.
    display_routing_results: list[RoutingResult] = []
    display_answer_cases: list[AnswerCase] = []
    all_score_dfs: list = []
    all_routing_results: list[list[RoutingResult]] = []

    overall_t0 = time.perf_counter()

    for run_idx in range(args.runs):
        if args.runs > 1:
            print()
            print("─" * 72)
            print(f"  RUN {run_idx + 1} of {args.runs}")
            print("─" * 72)

        run_routing_results: list[RoutingResult] = []
        run_answer_cases: list[AnswerCase] = []

        for category in sorted(by_category):
            print(f"[{category}]")
            for case in by_category[category]:
                routing, answer_case = run_and_capture(
                    case, suite, run_id,
                    run_idx=run_idx, total_runs=args.runs,
                )
                run_routing_results.append(routing)

                label = "PASS" if routing.passed else "FAIL"
                marker = "✓" if routing.passed else "✗"
                ragas_tag = " + captured for RAGAS" if answer_case is not None else ""
                print(
                    f"  {marker} [{label}] {case.id:<28} "
                    f"({routing.elapsed_s:.2f}s)  {routing.reason}{ragas_tag}"
                )
                if answer_case is not None:
                    run_answer_cases.append(answer_case)
            print()

        all_routing_results.append(run_routing_results)
        if run_idx == 0:
            display_routing_results = run_routing_results
            display_answer_cases = run_answer_cases

        if run_answer_cases:
            print(f"Scoring {len(run_answer_cases)} answer case(s) "
                  f"with RAGAS judge={args.judge_model}...")
            print("(Each case ≈ 3-9 judge LLM calls. This takes a minute.)")
            result = score_with_ragas(run_answer_cases, judge_model=args.judge_model)
            all_score_dfs.append(result.to_pandas())

    overall_elapsed_s = time.perf_counter() - overall_t0

    routing_results = display_routing_results
    answer_cases = display_answer_cases

    _print_routing_summary(all_routing_results, overall_elapsed_s)

    if not answer_cases:
        print("\nNo answer-producing cases — skipping RAGAS scoring.")
        if not args.no_report:
            path = write_markdown_report(
                routing_results=routing_results,
                all_routing_results=all_routing_results,
                overall_elapsed_s=overall_elapsed_s,
                answer_cases=[],
                scores_df=None,
                stds_df=None,
                n_runs=args.runs,
                judge_model=args.judge_model,
                note=args.note,
                run_id=run_id,
            )
            print(f"\n  Report written: {path.relative_to(PROJECT_ROOT)}")
        flush_traces()
        # Strictest exit code: any failure in any run is a non-zero exit. A flaky
    # case that passes 2 of 3 runs still counts as a failure for CI purposes —
    # routing is supposed to be deterministic, and intermittent failure means
    # something is wrong even if the "average" looks fine.
    return 0 if all(r.passed for run in all_routing_results for r in run) else 1

    means_df, stds_df = _aggregate_score_dfs(all_score_dfs)
    _print_ragas_aggregate(
        means_df, stds_df,
        n_cases=len(answer_cases),
        n_runs=args.runs,
        answer_cases=answer_cases,
    )

    if not args.no_langfuse_scores and settings.langfuse_enabled:
        # Attach the AVERAGED scores to the run-1 traces (the only ones whose
        # answer text is in the report). This gives a stable, comparable
        # dashboard score per case without N duplicates per metric.
        n = attach_scores_to_langfuse(answer_cases, means_df, judge_model=args.judge_model)
        print(f"\n  Attached {n} averaged score(s) to Langfuse traces.")
        print(f"  View session: https://cloud.langfuse.com/sessions/{run_id}")
    elif not settings.langfuse_enabled:
        print("\n  (Langfuse not configured — skipping score attachment.)")

    if not args.no_report:
        path = write_markdown_report(
            routing_results=routing_results,
            all_routing_results=all_routing_results,
            overall_elapsed_s=overall_elapsed_s,
            answer_cases=answer_cases,
            scores_df=means_df,
            stds_df=stds_df,
            n_runs=args.runs,
            judge_model=args.judge_model,
            note=args.note,
            run_id=run_id,
        )
        print(f"\n  Report written: {path.relative_to(PROJECT_ROOT)}")

    flush_traces()
    # Strictest exit code: any failure in any run is a non-zero exit. A flaky
    # case that passes 2 of 3 runs still counts as a failure for CI purposes —
    # routing is supposed to be deterministic, and intermittent failure means
    # something is wrong even if the "average" looks fine.
    return 0 if all(r.passed for run in all_routing_results for r in run) else 1


if __name__ == "__main__":
    sys.exit(main())
