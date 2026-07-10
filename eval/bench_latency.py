"""
bench_latency.py — per-stage latency benchmark over the eval test cases.

Runs each case in test_cases.yaml through the real run_chat() pipeline with
per-stage timing (backend.app.timing) and reports where the wall time goes:

    classify_message
    retrieve_and_rerank
        embed          (OpenAI embedding)
        pinecone       (3 hybrid source queries)
        rerank_load    (cross-encoder model load — cold start only)
        rerank_infer   (cross-encoder scoring of the candidate pool)
    answer_llm
    review_answer
    answerability
    validate_and_fix

This is the PERFORMANCE-layer counterpart to run_eval.py (routing correctness)
and ragas_eval.py (answer quality). It asserts nothing — it measures. Reuse it
to capture a before/after when tuning latency (warmup, parallel retrieval, etc).

Usage:
    python -m eval.bench_latency                       # all cases, 1 pass
    python -m eval.bench_latency --repeat 3 --warmup   # 3 warm passes (recommended)
    python -m eval.bench_latency --drop-first          # exclude the cold pass from stats
    python -m eval.bench_latency --case iron_basic
    python -m eval.bench_latency --category core_nutrition
    python -m eval.bench_latency --label "baseline (pre-optimization)"
    python -m eval.bench_latency --no-report
"""

import argparse
import statistics
import time
from datetime import datetime

from backend.app.chat.pipeline import run_chat
from backend.app.config import PROJECT_ROOT
from backend.app.models.schemas import ChatRequest
from backend.app.observability import flush as flush_traces
from backend.app.timing import collect_timings
from eval.run_eval import load_suite

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# Display order. Sub-stages of retrieve_and_rerank are shown indented beneath it.
TOP_STAGES = [
    "classify_message",
    "retrieve_and_rerank",
    "answer_llm",
    "review_answer",
    "answerability",
    "validate_and_fix",
]
SUB_STAGES = ["hyde", "embed", "pinecone", "rerank_load", "rerank_infer"]


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0, 1])."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stat_row(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": statistics.median(values) if values else 0.0,
        "p90": percentile(values, 0.9),
        "mean": statistics.mean(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def run(args: argparse.Namespace) -> None:
    suite = load_suite()
    cases = suite.cases
    if args.case:
        cases = [c for c in cases if c.id == args.case]
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if not cases:
        print("No matching cases.")
        return

    if args.warmup:
        print("Warming up the reranker (model load + kernel compile) ...", flush=True)
        from backend.app.rag.retriever import warmup_reranker

        warmup_reranker()

    runs: list[dict] = []
    print(f"\nRunning {len(cases)} case(s) × {args.repeat} pass(es)\n")
    print(f"  {'pass':<5}{'case':<28}{'response':<13}{'wall(s)':>8}")
    for r in range(args.repeat):
        for c in cases:
            req = ChatRequest(message=c.query, user_profile=suite.profiles[c.profile])
            with collect_timings() as timings:
                t0 = time.perf_counter()
                resp = run_chat(req)
                wall = time.perf_counter() - t0
            runs.append(
                {
                    "id": c.id,
                    "category": getattr(c.category, "value", c.category),
                    "response_type": resp.response_type.value,
                    "wall": wall,
                    "pass": r,
                    "stages": dict(timings),
                }
            )
            print(
                f"  {r + 1:<5}{c.id:<28}{resp.response_type.value:<13}{wall:>8.2f}",
                flush=True,
            )

    flush_traces()

    # Exclude the first (cold) pass from aggregation if asked and we have >1 pass.
    agg = [x for x in runs if not (args.drop_first and x["pass"] == 0)]
    if not agg:
        agg = runs

    report = _render(agg, args)
    print(report)

    if not args.no_report:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"bench_latency_{ts}.md"
        path.write_text(report)
        print(f"\nReport written to {path.relative_to(PROJECT_ROOT)}")


def _render(agg: list[dict], args: argparse.Namespace) -> str:
    # A "full-pipeline" run is one that actually reached the answer LLM (answer
    # or no_results); short-circuits (emergency / out_of_scope) only classify.
    full = [x for x in agg if "answer_llm" in x["stages"]]
    short = [x for x in agg if "answer_llm" not in x["stages"]]

    total_full = stat_row([x["wall"] for x in full])

    lines: list[str] = []
    lines.append("# Latency benchmark — per stage\n")
    if args.label:
        lines.append(f"**Label:** {args.label}\n")
    lines.append(
        f"- Runs: {len(agg)} total "
        f"({len(full)} full-pipeline, {len(short)} short-circuit)"
    )
    if args.drop_first and args.repeat > 1:
        lines.append("- First (cold) pass excluded from stats")
    if args.warmup:
        lines.append("- Reranker warmed before the run")
    lines.append("")

    # --- Per-stage table over full-pipeline runs ---
    lines.append("## Per-stage breakdown (full-pipeline answers)\n")
    lines.append("| stage | median | p90 | max | share |")
    lines.append("|---|---:|---:|---:|---:|")

    def row(name: str, indent: bool = False) -> None:
        vals = [x["stages"][name] for x in full if name in x["stages"]]
        if not vals:
            return
        s = stat_row(vals)
        share = (
            f"{100 * s['median'] / total_full['median']:.0f}%"
            if total_full["median"]
            else "—"
        )
        label = ("&nbsp;&nbsp;↳ " + name) if indent else f"**{name}**"
        lines.append(
            f"| {label} | {s['median']:.2f}s | {s['p90']:.2f}s | "
            f"{s['max']:.2f}s | {'' if indent else share} |"
        )

    for stage in TOP_STAGES:
        row(stage)
        if stage == "retrieve_and_rerank":
            for sub in SUB_STAGES:
                row(sub, indent=True)

    lines.append(
        f"| **TOTAL (wall)** | {total_full['median']:.2f}s | "
        f"{total_full['p90']:.2f}s | {total_full['max']:.2f}s | 100% |"
    )
    lines.append("")

    # --- Short-circuit summary ---
    if short:
        cs = stat_row([x["wall"] for x in short])
        lines.append(
            f"Short-circuit (emergency / out_of_scope) wall: "
            f"median {cs['median']:.2f}s, p90 {cs['p90']:.2f}s "
            f"(classify only, n={cs['n']}).\n"
        )

    # --- Per-case medians (wall) ---
    lines.append("## Per-case wall latency (median across passes)\n")
    lines.append("| case | category | response | median wall |")
    lines.append("|---|---|---|---:|")
    by_case: dict[str, list[dict]] = {}
    for x in agg:
        by_case.setdefault(x["id"], []).append(x)
    for cid, xs in by_case.items():
        med = statistics.median([x["wall"] for x in xs])
        lines.append(
            f"| {cid} | {xs[0]['category']} | {xs[0]['response_type']} | {med:.2f}s |"
        )

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-stage latency benchmark.")
    ap.add_argument("--repeat", type=int, default=1, help="passes over the case set")
    ap.add_argument("--warmup", action="store_true", help="warm the reranker first")
    ap.add_argument(
        "--drop-first",
        action="store_true",
        help="exclude the first (cold) pass from stats (needs --repeat > 1)",
    )
    ap.add_argument("--case", help="run only this case id")
    ap.add_argument("--category", help="run only this category")
    ap.add_argument("--label", help="note recorded in the report header")
    ap.add_argument("--no-report", action="store_true", help="don't write a markdown report")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
