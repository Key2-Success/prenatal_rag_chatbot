"""
run_eval.py — v1 evaluation runner for Poshan Saathi.

Loads `user_profiles.yaml` and `test_cases.yaml`, validates them through
the schemas in `eval/schemas.py`, runs each case through the real
`run_chat()` pipeline (Pinecone retrieval + OpenAI LLM), and asserts:

  1. behavior  — answer | emergency | out_of_scope | no_results
  2. cites_org — (optional, behavior=answer only) first source's org

Prints per-case PASS/FAIL inline, a per-category summary at the end, and
writes a timestamped markdown report to `eval/results/`.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --category core_nutrition
    python -m eval.run_eval --case iron_basic
    python -m eval.run_eval --no-report
    python -m eval.run_eval -m "raised threshold to 0.55"

Why a real-pipeline runner (not mocks):
  Behaviour, retrieval, and LLM output all interact. Mocking retrieval
  would hide exactly the bugs we want to catch (e.g. relevant query →
  fallback). It costs cents per run — fine for v1.
"""

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from backend.app.chat.pipeline import run_chat
from backend.app.config import PROJECT_ROOT
from backend.app.models.schemas import ChatRequest, ResponseType
from eval.schemas import EvalSuite, TestCase

EVAL_DIR = PROJECT_ROOT / "eval"
PROFILES_PATH = EVAL_DIR / "user_profiles.yaml"
CASES_PATH = EVAL_DIR / "test_cases.yaml"
RESULTS_DIR = EVAL_DIR / "results"

# Truncate the answer when echoing into the markdown report — full answers
# would make the report hundreds of lines and noisy to diff.
_PREVIEW_CHARS = 160


# ---------- Loading ----------

def load_suite() -> EvalSuite:
    """
    Parse both YAML files into a validated EvalSuite. Schema, enum, cross-
    field, and cross-document checks all happen here — by the time this
    returns, everything downstream can trust its inputs.
    """
    with open(PROFILES_PATH) as f:
        profiles_raw = yaml.safe_load(f)
    with open(CASES_PATH) as f:
        cases_raw = yaml.safe_load(f)
    return EvalSuite(profiles=profiles_raw, cases=cases_raw)


# ---------- Single-case execution ----------

@dataclass
class CaseResult:
    """Outcome of running one test case."""
    case: TestCase
    passed: bool
    reason: str
    elapsed_s: float
    actual_type: ResponseType | None
    actual_org: str | None
    answer_preview: str | None


def _evaluate(case: TestCase, response, elapsed: float) -> CaseResult:
    """Compare a successful pipeline response against the case's expectations."""
    actual_type = response.response_type
    actual_org = response.sources[0].org_display_name if response.sources else None

    failures: list[str] = []
    if actual_type is not case.expected.behavior:
        failures.append(
            f"behavior: expected {case.expected.behavior.value}, "
            f"got {actual_type.value}"
        )
    # Citation check fires only on answer-behavior cases (the schema guarantees
    # cites_org is None for non-answer cases).
    if (
        case.expected.cites_org
        and actual_type is ResponseType.answer
        and actual_org != case.expected.cites_org
    ):
        failures.append(
            f"cites_org: expected {case.expected.cites_org}, got {actual_org}"
        )

    return CaseResult(
        case=case,
        passed=not failures,
        reason="; ".join(failures) if failures else "ok",
        elapsed_s=elapsed,
        actual_type=actual_type,
        actual_org=actual_org,
        answer_preview=response.answer[:_PREVIEW_CHARS].replace("\n", " "),
    )


def run_case(case: TestCase, suite: EvalSuite) -> CaseResult:
    """Execute one validated test case end-to-end."""
    profile = suite.profiles[case.profile]  # safe: validator guarantees existence
    request = ChatRequest(message=case.query, user_profile=profile)

    t0 = time.perf_counter()
    try:
        response = run_chat(request)
    except Exception as e:
        return CaseResult(
            case=case,
            passed=False,
            reason=f"pipeline raised: {type(e).__name__}: {e}",
            elapsed_s=time.perf_counter() - t0,
            actual_type=None,
            actual_org=None,
            answer_preview=None,
        )
    return _evaluate(case, response, time.perf_counter() - t0)


# ---------- Console output ----------

def _print_case(result: CaseResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    marker = "✓" if result.passed else "✗"
    print(
        f"  {marker} [{status}] {result.case.id:<28} "
        f"({result.elapsed_s:.2f}s)  {result.reason}"
    )


def _print_summary(results: list[CaseResult]) -> None:
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    by_category = _group_by_category(results)
    for category in sorted(by_category):
        items = by_category[category]
        passed = sum(1 for r in items if r.passed)
        total = len(items)
        bar = "█" * passed + "░" * (total - passed)
        print(f"  {category:<20} {passed}/{total}  {bar}")

    total_passed = sum(1 for r in results if r.passed)
    total_time = sum(r.elapsed_s for r in results)
    print()
    print(f"  TOTAL  {total_passed}/{len(results)} passed  ({total_time:.1f}s)")


def _group_by_category(results: list[CaseResult]) -> dict[str, list[CaseResult]]:
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_category[r.case.category.value].append(r)
    return by_category


# ---------- Markdown report ----------

def _report_lines(results: list[CaseResult], note: str | None, timestamp: str) -> list[str]:
    """Build the markdown report content. Pure function — easy to test."""
    by_category = _group_by_category(results)
    total_passed = sum(1 for r in results if r.passed)
    total_time = sum(r.elapsed_s for r in results)

    lines: list[str] = [
        f"# Poshan Saathi — Eval Run {timestamp}",
        "",
        f"**Result:** {total_passed}/{len(results)} passed in {total_time:.1f}s",
        "",
    ]

    if note:
        # Run note: short message about what changed (e.g. "raised threshold
        # to 0.55"). Makes the results/ directory diff-friendly.
        lines += ["## Note", "", f"> {note}", ""]

    lines += ["## Per-category", "", "| Category | Passed | Total |", "|---|---|---|"]
    for category in sorted(by_category):
        items = by_category[category]
        passed = sum(1 for r in items if r.passed)
        lines.append(f"| {category} | {passed} | {len(items)} |")
    lines.append("")

    lines += [
        "## Cases",
        "",
        "| ID | Category | Status | Expected | Actual | Cites (exp → got) | Time | Reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        actual = r.actual_type.value if r.actual_type else "—"
        cites = f"{r.case.expected.cites_org or '—'} → {r.actual_org or '—'}"
        lines.append(
            f"| {r.case.id} | {r.case.category.value} | {status} "
            f"| {r.case.expected.behavior.value} | {actual} "
            f"| {cites} | {r.elapsed_s:.2f}s | {r.reason} |"
        )
    lines.append("")

    lines += ["## Answer previews", ""]
    for r in results:
        actual = r.actual_type.value if r.actual_type else "error"
        lines += [f"**{r.case.id}** — _{actual}_", "", f"> {r.answer_preview or '(no response)'}", ""]

    return lines


def write_markdown_report(results: list[CaseResult], note: str | None) -> Path:
    """Write a timestamped markdown report. Diffable across tuning runs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"eval_{timestamp}.md"
    path.write_text("\n".join(_report_lines(results, note, timestamp)))
    return path


# ---------- Entry point ----------

def _filter_cases(cases: list[TestCase], category: str | None, case_id: str | None) -> list[TestCase]:
    if category:
        cases = [c for c in cases if c.category.value == category]
    if case_id:
        cases = [c for c in cases if c.id == case_id]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the v1 evaluation suite against the live RAG pipeline."
    )
    parser.add_argument("--category", help="Run only cases in this category")
    parser.add_argument("--case", help="Run only the case with this id")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip writing the markdown report")
    parser.add_argument("-m", "--note",
                        help="Short message describing what changed since the last run "
                             "(e.g. 'raised threshold to 0.55'). Embedded in the report.")
    args = parser.parse_args()

    try:
        suite = load_suite()
    except ValidationError as e:
        # Pydantic gives field-level messages — surface them and exit before
        # spending money on OpenAI calls.
        print("Suite failed validation:\n")
        print(e)
        return 2

    cases = _filter_cases(suite.cases, args.category, args.case)
    if not cases:
        print("No cases match the given filters.")
        return 1

    print(f"Running {len(cases)} case(s)...")
    if args.note:
        print(f"Note: {args.note}")
    print()

    by_category: dict[str, list[TestCase]] = defaultdict(list)
    for c in cases:
        by_category[c.category.value].append(c)

    results: list[CaseResult] = []
    for category in sorted(by_category):
        print(f"[{category}]")
        for case in by_category[category]:
            result = run_case(case, suite)
            results.append(result)
            _print_case(result)
        print()

    _print_summary(results)

    if not args.no_report:
        path = write_markdown_report(results, note=args.note)
        print(f"\n  Report written: {path.relative_to(PROJECT_ROOT)}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
