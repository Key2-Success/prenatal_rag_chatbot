# Common pitfalls

The mistakes that bite teams when adopting RAGAS, and how to avoid them.

## 1. Cost runaway

**Symptom:** "I ran RAGAS on 1000 cases with 5 metrics and my OpenAI bill is $40."

**Why:** Each metric makes 3-10 LLM calls per sample. Five metrics × 1000 samples × ~5 calls each = ~25k judge-LLM calls. With `gpt-4o` at standard pricing, that's real money.

**Mitigations:**
- **Sample down for iteration.** When tuning your pipeline, run RAGAS on a representative 50-sample subset, not the full set.
- **Run the full eval on a schedule** (nightly, weekly), not on every PR. Use cheap behavioural tests as the per-PR gate.
- **Use a smaller judge for iteration, a stronger judge for the canonical run.** `gpt-4o-mini` is ~10x cheaper than `gpt-4o` and good enough to detect direction-of-change. Promote to `gpt-4o` only for the published number.
- **Cache embeddings.** If you re-run the same dataset multiple times, the embedding-based metrics (`ResponseRelevancy`, `SemanticSimilarity`) recompute the same vectors. RAGAS doesn't cache these by default — wrap your embedder in a memoising layer if iterating.

Approximate budgeting: assume ~$0.001-0.005 per (sample × metric) with `gpt-4o-mini`, and ~10x that with `gpt-4o`. Multiply by your dataset size to estimate before running.

## 2. Judge bias from same-model judging

**Symptom:** "All my answers score above 0.9 on every metric, but real users say the system is mediocre."

**Why:** A model judging its own output systematically inflates scores. This is the single biggest threat to RAGAS scores being meaningful.

**Mitigations:**
- The judge **must** be a different model from your answer LLM. Different vendor is even better (e.g. answer LLM is OpenAI, judge is Claude).
- The judge should be **at least as strong** as the answer LLM. Otherwise it can't reliably catch the answer's errors.
- Run a small **calibration set** of known-bad answers (intentional hallucinations, off-topic responses) and confirm the judge scores them low. If a known-bad answer gets >0.7 on faithfulness, your judge is too generous — try a stronger model or a more specific prompt template.

## 3. Dataset format mistakes

**Symptom:** Cryptic errors at `evaluate()` time, or all scores come back NaN.

The field names RAGAS expects are precise — typos and casing matter:

```python
# CORRECT
{
    "user_input": "...",
    "response": "...",
    "retrieved_contexts": ["chunk1", "chunk2"],   # list of strings
    "reference": "...",                            # only for ref-required metrics
}

# COMMON MISTAKES
{
    "query": "...",                # WRONG — RAGAS expects "user_input"
    "answer": "...",               # WRONG — RAGAS expects "response"
    "retrieved_contexts": "...",   # WRONG — must be a LIST of strings, not one string
    "contexts": [...],             # WRONG — old field name, now "retrieved_contexts"
}
```

Always build the dataset via `EvaluationDataset.from_list([...])` rather than passing raw lists/dicts to `evaluate()` directly — the wrapper validates field shape and gives clearer errors.

## 4. Treating RAGAS as a routing/behavioural test

**Symptom:** "I want RAGAS to confirm my emergency-detection guardrail fired correctly."

**Why this is wrong:** RAGAS scores *answer content quality*. It doesn't know about your pipeline's routing, classification, or guardrails. If your pipeline produced an emergency redirect (canned text), RAGAS will happily score "faithfulness" on it but the score is meaningless because there's no retrieval to be faithful to.

**Mitigations:**
- Keep behavioural tests separate. Pipeline-routing assertions belong in unit/integration tests, not in RAGAS.
- Filter out non-`answer` cases before running RAGAS. Only score cases where the pipeline actually produced a retrieval-grounded answer.

## 5. Score volatility / "the same eval gives different numbers"

**Symptom:** Run RAGAS twice on the same dataset, scores differ by 0.05-0.10.

**Why:** Judge LLM calls are stochastic by default. Even with `temperature=0`, sampling can vary at the API level.

**Mitigations:**
- Always pin `temperature=0` on the judge.
- For canonical runs, compute scores as the mean of N=3 runs and report `mean ± stddev`. Single-run numbers are noisy.
- Use the same judge model+version across comparison runs. A "gpt-4o" upgrade silently shifting versions can move scores by 0.05+.

## 6. Comparing scores across different metric versions

**Symptom:** "We upgraded RAGAS from 0.1.x to 0.2.x and our faithfulness scores all dropped."

**Why:** RAGAS sometimes changes the metric prompts or scoring formulas between versions. The numbers aren't directly comparable.

**Mitigations:**
- Pin RAGAS to a specific minor version (`ragas>=0.2,<0.3`) for any longitudinal comparison.
- When upgrading, re-baseline. Document the upgrade in your eval reports so future readers know not to compare across the boundary.

## 7. Letting RAGAS replace human review

**Symptom:** "Faithfulness is 0.92, ship it."

**Why this is wrong:** LLM-as-judge metrics correlate with human quality but are not a substitute. Judges can miss subtle factual errors a human would catch, and they reward fluency in ways that don't always map to correctness.

**Mitigations:**
- Use RAGAS for *trends* (is this PR better or worse than baseline?) and *outliers* (which 5 cases scored below 0.6?), not as a binary pass/fail gate without human spot-checks.
- Pair RAGAS reports with a sample of low-scoring cases that a human reviewed. If the human disagrees with the judge often, your judge prompt or model is wrong.
