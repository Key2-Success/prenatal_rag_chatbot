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
) -> tuple[RoutingResult, AnswerCase | None]:
    """
    Run one test case end-to-end.

    Always returns a RoutingResult (all 26 cases are routing-evaluated).
    Also returns an AnswerCase when the pipeline produced response_type=answer;
    returns None in that slot for emergency / out_of_scope / no_results because
    RAGAS metrics are meaningless on canned text.

    run_id groups all traces from one eval run into a single Langfuse
    Session so you can see every case + its RAGAS scores in one view.
    """
    profile = suite.profiles[case.profile]
    request = ChatRequest(message=case.query, user_profile=profile)
    capture: dict = {}

    t0 = time.perf_counter()
    try:
        with propagate_attributes(
            trace_name=case.id,
            session_id=run_id,
            tags=["ragas_eval", case.category.value],
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


def _print_routing_summary(routing_results: list[RoutingResult]) -> None:
    print()
    print("=" * 72)
    print("  ROUTING SUMMARY (all cases)")
    print("=" * 72)

    by_category = _group_by_category(routing_results)
    for category in sorted(by_category):
        items = by_category[category]
        passed = sum(1 for r in items if r.passed)
        total = len(items)
        bar = "█" * passed + "░" * (total - passed)
        print(f"  {category:<25} {passed}/{total}  {bar}")

    total_passed = sum(1 for r in routing_results if r.passed)
    total = len(routing_results)
    total_time = sum(r.elapsed_s for r in routing_results)
    print()
    print(f"  TOTAL  {total_passed}/{total} passed  ({total_time:.1f}s)")


def _print_ragas_aggregate(scores_df, n_cases: int, answer_cases: list[AnswerCase]) -> None:
    print()
    print("=" * 72)
    print("  RAGAS AGGREGATE SCORES (answer cases only)")
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


# ---------- Markdown report ----------

def _markdown_report(
    routing_results: list[RoutingResult],
    answer_cases: list[AnswerCase],
    scores_df,
    judge_model: str,
    note: str | None,
    timestamp: str,
) -> str:
    n_total = len(routing_results)
    n_routing_passed = sum(1 for r in routing_results if r.passed)
    n_scored = len(answer_cases)
    total_time = sum(r.elapsed_s for r in routing_results)
    routing_pass_rate = (n_routing_passed / n_total * 100) if n_total else 0.0

    lines: list[str] = [
        f"# Poshan Saathi — Eval {timestamp}",
        "",
        f"**Total time:** {total_time:.1f}s  "
        f"**Judge model:** `{judge_model}` (temperature 0)",
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
        f"**{n_routing_passed}/{n_total} passed ({routing_pass_rate:.0f}%)** "
        "across all cases. Checks `response_type` matches expected behavior, "
        "and (for answer cases) the first cited source matches `cites_org`.",
        "",
        "| Category | Passed | Total |",
        "|---|---|---|",
    ]
    by_category = _group_by_category(routing_results)
    for category in sorted(by_category):
        items = by_category[category]
        passed = sum(1 for r in items if r.passed)
        lines.append(f"| {category} | {passed} | {len(items)} |")
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
        lines += [
            f"**{n_scored} answer case(s)** scored on three reference-free "
            "metrics. Higher is better; 1.0 is the ceiling.",
            "",
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
                lines.append(
                    f"| {metric} | {mean:.3f} | {metric_descriptions[metric]} |"
                )
        lines.append("")

    # ── Section 1: Routing breakdown ─────────────────────────────────────────
    lines += [
        "## Routing breakdown",
        "",
        "Per-case detail for all cases — useful when a category in the summary "
        "above is below 100% and you need to find the failing case quickly.",
        "",
        "| ID | Category | Status | Expected | Actual | Cites (exp → got) | Time | Reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in routing_results:
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
        for ac, (_, row) in zip(answer_cases, scores_df.iterrows()):
            lines.append(
                f"| {ac.case.id} | {ac.case.category.value} "
                f"| {row.get('faithfulness', float('nan')):.3f} "
                f"| {row.get('answer_relevancy', float('nan')):.3f} "
                f"| {row.get('llm_context_precision_without_reference', float('nan')):.3f} "
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

    routing_results: list[RoutingResult] = []
    answer_cases: list[AnswerCase] = []

    by_category: dict[str, list[TestCase]] = defaultdict(list)
    for c in cases:
        by_category[c.category.value].append(c)

    for category in sorted(by_category):
        print(f"[{category}]")
        for case in by_category[category]:
            routing, answer_case = run_and_capture(case, suite, run_id)
            routing_results.append(routing)

            label = "PASS" if routing.passed else "FAIL"
            marker = "✓" if routing.passed else "✗"
            ragas_tag = " + captured for RAGAS" if answer_case is not None else ""
            print(
                f"  {marker} [{label}] {case.id:<28} "
                f"({routing.elapsed_s:.2f}s)  {routing.reason}{ragas_tag}"
            )
            if answer_case is not None:
                answer_cases.append(answer_case)
        print()

    _print_routing_summary(routing_results)

    if not answer_cases:
        print("\nNo answer-producing cases — skipping RAGAS scoring.")
        if not args.no_report:
            path = write_markdown_report(
                routing_results=routing_results,
                answer_cases=[],
                scores_df=None,
                judge_model=args.judge_model,
                note=args.note,
                run_id=run_id,
            )
            print(f"\n  Report written: {path.relative_to(PROJECT_ROOT)}")
        flush_traces()
        return 0 if all(r.passed for r in routing_results) else 1

    print(f"\nScoring {len(answer_cases)} answer case(s) with RAGAS judge={args.judge_model}...")
    print("(Each case ≈ 3-9 judge LLM calls. This takes a minute.)")
    print()

    result = score_with_ragas(answer_cases, judge_model=args.judge_model)
    df = result.to_pandas()
    _print_ragas_aggregate(df, n_cases=len(answer_cases), answer_cases=answer_cases)

    if not args.no_langfuse_scores and settings.langfuse_enabled:
        n = attach_scores_to_langfuse(answer_cases, df, judge_model=args.judge_model)
        print(f"\n  Attached {n} score(s) to Langfuse traces.")
        print(f"  View session: https://cloud.langfuse.com/sessions/{run_id}")
    elif not settings.langfuse_enabled:
        print("\n  (Langfuse not configured — skipping score attachment.)")

    if not args.no_report:
        path = write_markdown_report(
            routing_results=routing_results,
            answer_cases=answer_cases,
            scores_df=df,
            judge_model=args.judge_model,
            note=args.note,
            run_id=run_id,
        )
        print(f"\n  Report written: {path.relative_to(PROJECT_ROOT)}")

    flush_traces()
    return 0 if all(r.passed for r in routing_results) else 1


if __name__ == "__main__":
    sys.exit(main())
