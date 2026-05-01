---
name: ragas
description: Use this skill whenever the user wants to evaluate, score, benchmark, or measure the quality of a RAG (Retrieval-Augmented Generation) pipeline — including faithfulness, answer relevancy, context precision/recall, hallucination detection, or "how good are my RAG answers?". Trigger on any mention of RAGAS, RAG evaluation, LLM-as-judge for RAG, evaluating retrieval quality, scoring generated answers against retrieved contexts, or comparing RAG configurations. Also use when wiring RAGAS scores into Langfuse traces, or when the user asks "what should I use to evaluate my RAG system?". This skill picks the right metrics for the user's situation (with or without ground-truth references), wires up the judge LLM correctly, and keeps evaluation costs in check.
---

# RAGAS

This skill teaches an AI assistant how to evaluate RAG pipelines with [RAGAS](https://docs.ragas.io) — the open-source LLM-as-judge framework by ExplodingGradients. RAGAS scores the *content quality* of generated answers (faithfulness, relevance, retrieval precision), which is orthogonal to behavioural/routing tests that check whether the right pipeline path fired.

## Core Principles

Follow these for ALL RAGAS work:

1. **Documentation first**: NEVER write RAGAS code from memory. The library has renamed metrics (e.g. `AnswerRelevancy` → `ResponseRelevancy`) and refactored namespaces. Always check the [current docs](https://docs.ragas.io/) before writing imports.
2. **Pick metrics based on what data you have**, not what sounds impressive. If there are no reference (ground-truth) answers, use the reference-free subset. See `references/metrics.md`.
3. **Use a stronger judge than the answer LLM.** Judging with the same model that generated the answer creates self-favouring bias. A small but stronger judge (`gpt-4o-mini` for nano-tier answers, `gpt-4o` for mini-tier answers) keeps scoring honest. See `references/judge-config.md`.
4. **Control cost up front.** RAGAS makes 3-10 LLM calls per (sample × metric). On a 100-case eval with 4 metrics, that's 1200–4000 calls. Sample down or pin a budget before running on a large dataset. See `references/pitfalls.md`.
5. **Layer with behavioural eval, don't replace it.** RAGAS evaluates *answer content*. It does NOT replace tests that check routing, classification, or pipeline structure. Run both.

## Use-case-specific references

- Installing RAGAS and its peer deps: `references/installation.md`
- Choosing which metrics to use (with/without ground truth): `references/metrics.md`
- Configuring the judge LLM and embeddings: `references/judge-config.md`
- Sending RAGAS scores back to Langfuse traces: `references/integration-langfuse.md`
- Cost control, judge bias, and dataset format pitfalls: `references/pitfalls.md`

## Quick-start: a minimal reference-free RAG eval

Use this as the canonical starting point. It works without any ground-truth answers and uses the three reference-free metrics that match how most teams describe RAG quality. Adapt the judge model to be stronger than the answer model in your project.

```python
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,                          # answer grounded in retrieved chunks
    ResponseRelevancy,                     # answer addresses the question
    LLMContextPrecisionWithoutReference,   # retrieved chunks were relevant
)

# 1. Build the dataset. Field names are fixed by RAGAS — don't rename.
samples = [
    {
        "user_input": "How much iron should I take daily?",
        "response":   "60 mg of elemental iron, per the guidelines you shared.",
        "retrieved_contexts": [
            "Pregnant women should consume 60 mg of elemental iron and 500 mcg folic acid daily.",
            "Iron supplementation reduces anaemia risk during the second and third trimester.",
        ],
        # "reference": "<ground-truth answer>",  # only if you have one
    },
    # ... more samples
]
dataset = EvaluationDataset.from_list(samples)

# 2. Wrap a stronger model than the answer LLM as the judge.
judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))

# 3. Evaluate. result has aggregate + per-row scores.
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()],
    llm=judge,
)

print(result)                 # aggregate scores: {'faithfulness': 0.87, ...}
df = result.to_pandas()       # per-row scores, one row per sample
```

## Decision tree: which metrics?

```
Do you have ground-truth reference answers for each query?
├── No (most projects, especially early-stage)
│   └── Use: Faithfulness + ResponseRelevancy + LLMContextPrecisionWithoutReference
│       — covers "answer grounded?", "answer relevant?", "retrieval relevant?"
│
└── Yes (you've curated a golden dataset)
    └── Add: LLMContextRecall + FactualCorrectness
        — covers "did retrieval find everything?" and "factual vs reference?"
```

For the full taxonomy and selection logic see `references/metrics.md`.

## Layering RAGAS on existing evals

Most projects already have **behavioural** tests — assertions like "this query should return `response_type=answer`" or "this should hit the emergency path". RAGAS does NOT replace those. The two layers answer different questions:

| Layer        | Question                                       | Tool             |
| ------------ | ---------------------------------------------- | ---------------- |
| Behavioural  | Did the pipeline route the request correctly?  | unit/integration |
| Answer-quality (this skill) | Was the answer content actually good?     | RAGAS            |

Run them as two stages of the same eval suite — behavioural first (fast, deterministic, fails the build on regression), then RAGAS on cases that produced an answer (slow, stochastic, gates on aggregate score thresholds).

## Skill Feedback

If the user reports that this skill gave wrong instructions or missed a use case (e.g., RAGAS released a new metric the skill doesn't cover, or an import path here is stale because of a RAGAS API change), the right response is to (1) verify against current `docs.ragas.io`, (2) update the relevant reference file, and (3) note the date of the update in a comment. Do NOT trigger this for issues with RAGAS itself — only for issues with this skill's instructions.
