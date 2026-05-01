# Choosing the right metrics

The single most important question: **do you have ground-truth reference answers for each query?** RAGAS metrics split sharply along that axis.

## Reference-free metrics (no ground truth needed)

Use these when you have only `(user_input, response, retrieved_contexts)` — i.e. queries and what the system produced, but no curated "correct" answer to compare against. This is the common case for early-stage RAG projects.

| Metric                                  | Import                                                               | What it measures                                                                                                |
| --------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Faithfulness**                        | `from ragas.metrics import Faithfulness`                             | Are all factual claims in the answer supported by the retrieved contexts? (hallucination detector)              |
| **ResponseRelevancy**                   | `from ragas.metrics import ResponseRelevancy`                        | Does the answer address the question, or does it ramble / miss the point?                                       |
| **LLMContextPrecisionWithoutReference** | `from ragas.metrics import LLMContextPrecisionWithoutReference`      | Of the retrieved chunks, what proportion were actually relevant to the query? (retrieval precision proxy)       |
| **NoiseSensitivity**                    | `from ragas.metrics import NoiseSensitivity`                         | How much does the answer degrade when irrelevant chunks are mixed into the retrieval? (robustness)              |
| **ContextEntitiesRecall**               | `from ragas.metrics import ContextEntitiesRecall`                    | Did retrieval find the entities the question is about? (entity-coverage proxy when no reference exists)         |

**Default starter pack: `Faithfulness + ResponseRelevancy + LLMContextPrecisionWithoutReference`.** These three answer "is the answer grounded?", "is the answer relevant?", "was retrieval relevant?" — covering the three failure modes most reviewers care about. Add `NoiseSensitivity` if you specifically care about robustness to retrieval noise.

## Reference-required metrics (need a curated golden answer)

Use these only when you have a `reference` field — a curated ground-truth answer for each query. Building this dataset is real work; don't pretend you have one.

| Metric                              | Import                                                         | What it measures                                                                          |
| ----------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **LLMContextRecall**                | `from ragas.metrics import LLMContextRecall`                   | Did retrieval find ALL the information needed for the reference answer? (recall)          |
| **LLMContextPrecisionWithReference** | `from ragas.metrics import LLMContextPrecisionWithReference`  | Stricter context-precision that uses the reference to identify what should be relevant.   |
| **FactualCorrectness**              | `from ragas.metrics import FactualCorrectness`                 | Are the facts in the answer correct *vs the reference*? (precision/recall over claims)    |
| **AnswerCorrectness**               | `from ragas.metrics import AnswerCorrectness`                  | Combined score: factual correctness + semantic similarity to the reference.               |

If you have references, the "full" eval set is usually `Faithfulness + ResponseRelevancy + LLMContextPrecisionWithReference + LLMContextRecall + FactualCorrectness`.

## Building reference data — when is it worth it?

Curating a reference set is expensive (a domain expert writes the ideal answer for each query). It's worth doing when:

- **Regressions matter more than ceiling.** With references you can detect "model upgrade made things worse" at the *content* level, not just behavioural.
- **The domain has objectively right answers** (medical dosages, legal citations, math). Subjective domains don't benefit much.
- **You're past the prototyping phase.** Early on, reference-free metrics catch the obvious failures. Curate references after the obvious fires are out.

If you do this, store references in the same format RAGAS expects (`reference` field per sample), so the same eval script works regardless of which metrics you choose.

## What about classic NLP metrics? (BLEU, ROUGE, etc.)

RAGAS exposes these (`from ragas.metrics import BleuScore, RougeScore`). They're cheap (no LLM call) but correlate poorly with answer quality for RAG — they reward lexical overlap, not factuality. Use only as a sanity check or a fast proxy in CI, not as your primary metric.

## Note on metric renames

RAGAS has gone through metric renames. If you see code in old tutorials using these names, they're aliases or older names — prefer the current ones above:

| Old name (do not use) | Current name                          |
| --------------------- | ------------------------------------- |
| `AnswerRelevancy`     | `ResponseRelevancy`                   |
| `ContextPrecision`    | `LLMContextPrecisionWithoutReference` (or WithReference) |
| `ContextRecall`       | `LLMContextRecall`                    |
| `AnswerSimilarity`    | use `AnswerCorrectness` or `SemanticSimilarity` |

Verify against `docs.ragas.io` if you're unsure — the library is in active flux.
