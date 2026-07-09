# Poshan Saathi — Architecture History

*A prenatal-nutrition RAG chatbot for pregnant women in India (FastAPI · Pinecone · OpenAI · RAGAS).*

This document reconstructs, exhaustively and in order, every architectural decision, tuning
experiment, and dead-end from the project's first day (2026-04-09) to the present
(2026-06-30). It is assembled from three independent sources cross-checked against each other:

1. **The full development-chat transcript** (~458K tokens), mined in six time-sliced passes.
2. **The git history** — 54 commits on `main`, push-directly-to-main workflow.
3. **The eval-report archive** — 92 timestamped RAGAS runs in `eval/results/`, each carrying
   the tuning note (`-m`) that motivated it and its aggregate faithfulness / answer_relevancy /
   context_precision scores.

The intent is a complete record first, a narrative second. Dead-ends are kept, not pruned — the
reverted experiments (HyDE, the first LlamaParse migration, BM25-alone, MMR, temperature 0.1) are
as much a part of the story as the changes that shipped, and several are the strongest portfolio
signal precisely because they were measured and abandoned.

---

## Part I — The narrative arc

Six journeys run the length of the project. Each is a loop of *change → measure → keep or revert*,
and the honest reversals are load-bearing.

**The parser journey.** pypdf (flattens tables to "soup") → LlamaParse (cleaner markdown tables)
→ **reverted to pypdf** when answer quality dropped — then a git-bisect proved the drop was a
*prompt* commit, not the parser (a confounded-variable error, caught and corrected) → LlamaParse
**re-migrated deliberately**, this time with a measured A/B (context_precision 0.828 → 0.882).

**The reranker journey.** A hard source waterfall (MoHFW always beats WHO regardless of relevance)
→ a two-stage cross-encoder rerank with `bge-reranker-v2-m3`, hosted on Pinecone Inference → hit
the 500-calls/month free limit → **self-hosted the same model** via sentence-transformers on Apple
MPS (778ms for 15 pairs, zero rate limit). A Cohere "SOTA" recommendation was walked back — the
constraint was hosting economics, not model quality.

**The judge journey.** `gpt-4o-mini` (same tier as the answer model — too weak) → `gpt-4o` (hit a
429 TPM wall) → **cross-vendor `claude-sonnet-4-5-20250929`**, to avoid same-family self-favoring
bias (a durable rule: the judge must be at least as strong as the answer model, and ideally a
different vendor).

**The chunking journey.** Fixed 600-char `RecursiveCharacterTextSplitter` → `SemanticChunker`
(cut where embedding distance between sentence groups spikes) → percentile threshold 95 → 85 with
a 512-token hard-cap backstop → **header-aware chunking** (`MarkdownHeaderTextSplitter` + breadcrumb
prefix). Chunk count tracked the refinement: 307 → 494 → 903.

**The retrieval-signal journey.** Pure dense cosine → **hybrid dense + BM25** (`alpha=0.75`), to
surface rare Indian-context keywords (amla, ragi, jaggery). This forced the index metric from
cosine to dotproduct and a two-pass ingest (fit BM25 IDF on the full corpus first), and it exposed
that `similarity_threshold=0.3` was calibrated for cosine and mis-fires on the compressed hybrid
score range. HyDE was tried here too and abandoned — a strong reranker already does the work HyDE
targets, and profile-personalized hypotheticals dilute the query embedding.

**The faithfulness endgame.** The persistent problem: faithfulness stuck ~0.70–0.86 while the other
two metrics cleared 0.90. The root causes were a small model (gpt-4.1-nano/mini) inventing causal
wrap-ups, merging separate context items, and turning "for at least 180 days" into "daily." Soft
prompt rules kept getting bypassed even at temperature 0. The resolution was a philosophy shift:
**enforce deterministically what can be detected deterministically** — a post-generation
`review_answer` pass that decomposes into atomic claims and drops the unsupported ones, plus regex
strippers for deflective openers, embedded hedges, cadence riders, and answerability. The last
lever was diagnostic, not corrective: single-run faithfulness is *variance-dominated* (the same
deterministic answer scores 0.0–0.78 across judge runs), so temperature went 0.3 → 0 to remove
answer-side regeneration noise, and the canonical number is a multi-run average.

Two meta-themes recur throughout, both user-driven directives that shaped the engineering culture
of the repo:

- **Battle-tested libraries over custom** (RAGAS over a hand-rolled judge; SemanticChunker over
  ad-hoc splitting).
- **Recalibrate, don't remove** — when a threshold mis-fired, the wrong fix was deleting the
  filter ("BE A SENIOR AI ENGINEER… NO filtering at the first layer" was a *rejection* of setting
  the threshold to 0.0); the right fix was re-calibrating it for the new score distribution.

---

## Part II — Phase-by-phase inventory

### Phase 1 — Genesis & foundation (2026-04-09 → 05-01)

**Origin.** Productionizing a Colab LlamaIndex prototype: 3 credible PDFs (MoHFW, FOGSI, WHO),
user onboarding, out-of-scope fallbacks, emergency guardrails, a 0.7 cosine threshold, and a
priority-ordered per-PDF query. Goal: a deployable API and an AI-engineer portfolio piece using
"proper" RAG technique.

**Initial stack, with the rationale for each choice.**
- **FastAPI** over Flask/Django — pure async API, Pydantic validation, `/docs` Swagger, industry
  standard for ML backends. (Litestar/Starlette/Tornado/Express rejected.)
- **Pinecone**, single index filtered by metadata — the `knowledge_base_dictionary` was kept as
  `sources.json` config stamped onto each chunk's metadata rather than a separate store.
- **OpenAI `text-embedding-3-small`** (1536-dim) — 3-large was ~2× quality at ~7× cost;
  all-MiniLM local was the cheap fallback.
- **Answer LLM `gpt-4.1-nano`**, temperature 0.3.
- **Chunking:** `RecursiveCharacterTextSplitter`, 600 char / 100 overlap, min-length 50, pypdf.
- **Source priority** MoHFW (India gov) → FOGSI (India professional) → WHO (global), stored as
  `doc_reference_order`. Retrieval was a strict waterfall — a FOGSI 0.65 lost to a MoHFW 0.51.

**The config / path refactor saga (04-20).** A brittle `Path(__file__).parents[N]` count was
off-by-one; the fix escalated into `config.py` (pydantic-settings), then a `pyproject.toml` marker
+ `_find_project_root()` tree-walk (the pattern pytest/black/ruff use) so the root is
location-independent. A string of small mistakes (parents count, a dropped `from pathlib import
Path` causing a NameError) prompted a switch to Opus for a full codebase audit, which fixed real
bugs: the **enum f-string bug** (`f"{diet_type}"` rendered `"DietType.vegetarian"` into the prompt
instead of `"Vegetarian"` — fixed with `.value`), a hardcoded `SOURCE_PRIORITY_ORDER` derived from
sources.json instead, a new `sources.py` single-owner loader, and pip-installability via
`pyproject.toml`.

> **From my notes:** ensure config settings are tunable from ONE location / file settings. No overriding inside the class, but on a file level.

**Eval v1 — behavioral / routing (04-28).** A three-layer eval model was articulated (retrieval /
generation / pipeline-behavior) with a phased methodology (v1 deterministic, v2 LLM-judge, v3
RAGAS). Built: `eval/user_profiles.yaml` (4 personas — priya=Vegetarian/wk20/low-iron,
anjali=Non-Veg/wk8, meera=Vegetarian/wk32/hypertension+diabetes, sunita=Ovo-Vegetarian/wk12),
`eval/test_cases.yaml`, and `eval/run_eval.py` (runs the *real* `run_chat()`, asserts
`response_type`). The `behavior` enum has **four** values (answer / emergency / out_of_scope /
no_results) — deliberately splitting "fallback" into a guardrail path vs a retrieval-empty path.
Cases grew 20 → 23 as unanswerable topics were removed (anganwadi, ghee, chai — no doc covers them)
and personalization expanded to 3 conditions × 3 diets. A schema-first refactor made every module
boundary a Pydantic model (`Source`, `Chunk`, `RetrievedChunk`, `ChatResponse` with a
`ResponseType` enum), added a `clients.py` OpenAI singleton, and moved retrieval knobs into
env-tunable `Settings`.

> **From my notes:** planned test cases before having it written. This also helped decide where to optimize: experimented with different similarity_thresholds. Found out the above issues ^ as well.

> **From my notes:** the pipeline shape I kept coming back to — classify message / retrieve ordered chunks / answer with llm.

**LLM classifier migration (04-28).** Even a word-boundary regex flagged *"…keep my blood sugar in
check"* as an emergency. The fix replaced keyword guardrails entirely with an LLM triage call
(`classifier.py`, 3-way in_scope/emergency/out_of_scope, `gpt-4.1-nano` temp 0, fail-open) before
retrieval. `guardrails.py` was gutted to canned strings only. Three new eval cases covered exactly
what regex couldn't. This is where the **plan-first-confirm** directive was established (the
assistant proceeded without confirming four open decisions).

> **From my notes:** moving from predefined guardrails / out of scope to LLM checking it: in that way "blood sugar" in the message is not considered an emergency even though it has "blood" and real emergencies that don't have trigger words are also caught as emergencies. Reduces false positives and false negatives. An additional financial cost and latency to run an LLM model instead of a hardcoded list, but trade-off worth it in healthcare.

**Langfuse observability (04-29).** Wired against the v2 API, then the installed Langfuse skill's
"documentation-first" principle revealed the current SDK is v4 (OTel-based) — **reverted v2 → v4**,
rewrote the shim (`observe`, `update_current_span`, `propagate_attributes`, `flush`),
no-ops-when-unconfigured. Trace tree: `chat → {classify_message, retrieve → embeddings, answer_llm
→ create}`. A protracted 401 debug traced to US-vs-EU cloud host + a `LANGFUSE_BASE_URL`
(vs `_HOST`) typo silently ignored by `extra="ignore"`.

> **From my notes:** observability — latency for each section (average latency / cost in each section). Where things are failing in realtime / what do the inputs / outputs look like.

**RAGAS decision — a dead-end reversed twice (04-29 → 05-01).** The first proposal was a *custom*
5-metric judge. User correction #1: that conflates the two eval layers (personalization/safety are
already routing cases) — dropped to 3 RAGAS-aligned metrics. User correction #2: *"why reinvent the
wheel when RAGAS is more thoroughly tested… I am frustrated you are suggesting custom."* → pivoted
fully to the RAGAS library, and authored a reusable `.claude/skills/ragas/` skill (SKILL.md + 5
references). Two durable rules born here: prefer battle-tested libraries; audit the existing eval
structure before proposing new layers. `eval/ragas_eval.py` scores answer cases with Faithfulness +
ResponseRelevancy + LLMContextPrecisionWithoutReference (reference-free trio) and attaches scores
to Langfuse. Judge started at `gpt-4o-mini`.

> **From my notes:** evaluation helped me see how much hallucination I had.

> **From my notes:** but how good is even my answer evaluator? The answers seem relevant??

> **From my notes:** metrics of runs after each change. In readme —> FUTURE, CLEANER PLACE TO MONITOR THEM LOL.

### Phase 2 — Retrieval overhaul (2026-05-01 → 05-21)

**Two-stage retrieval replaces the waterfall.** `retrieve_ordered()` (hard MoHFW→FOGSI→WHO) →
`retrieve_and_rerank()`: Stage 1 recall (query all 3 sources, pool, dedup), Stage 2 cross-encoder
rerank (`bge-reranker-v2-m3`), Stage 3 order by (priority ASC, −score DESC) for context placement.
Dedup was corrected from `(doc_title, page)` to a **text MD5 hash** (two distinct chunks can share
a page). Additive source-priority score nudges (+0.02/+0.01) were rejected as untunable
hyperparameters — source preference lives only in context *ordering*. `reranker_candidate_k`
started at 5 (user-directed down from 10); `top_k` went 5 → 3 to shrink the "lost-in-the-middle"
window.

> **From my notes:** best way to debug is to go deep into your architecture / observability layer to see what is even happening: what are the responses and the intermediate steps — do they make sense?
>
> - sometimes, the top_k is returning the same retrieved chunks back. I want all 5 to be unique.
> - I am also seeing that right now, all the retrieved contexts are from MoHFW. This means that a 0.4 score of MoHFW will always trump a 0.7 score from FOGSI. Instead, I want to grab the top k from all 3 sources, order them by probability score within each source, and then grab a more refined set of top k. Would this be considered reranking? I would like to add a reranker into my architecture. Help me plan how to do this.
> - problem 1 — what if there are 2 different and highly relevant chunks on the same page on the same source?
> - and I don't want there to be manual nudges — that would be something I would need to tune as well. Why not order by source, score. Help me plan how I would architect in a way that would still prioritize source without manual nudges / numbers to tune — something more clever algorithmically.
> - candidate pool does not have to increase from 5 to 30 on first pass… not only more expensive but also our evaluation doesn't necessitate that yet. That's a 3:1 recall-to-output ratio, which is the standard starting point (from 15 to 5… look into that, is that true).

> **From my notes:** okay. Just to make sure that there is a probability of WHO chunks actually coming through: after the 15 chunks are chosen, then we choose the top 5 raw scores, and then we sort by the source, correct? And in doing this, we are inferring that there is more power the higher the chunk position it is? Is that true though, I thought llms care most about the first and last spots. —> Liu et al. 2023, "Lost in the Middle".

> **From my notes:** actually, I want to see how many of the top 5 scores are from what source and what its score is. Can you add this to the evaluation readme so I can evaluate source diversity and score from the answers? I will test what top_k = 5 feels like, evaluation wise, and then see if top_k = 3 is needed.

> **From my notes:** DIVERSITY GOOD if all the scores are high and they are competing for spots… but if WHO has terrible scores, then don't want to bring in diversity for the sake of diversity.

> **From my notes:** top_k = 3 was comparable (if not slightly better) to top_k = 5, so I kept 3 instead to save costs and latency.

**SemanticChunker replaces fixed chunking.** Root cause of low rerank scores (0.04–0.13) was
diagnosed as coarse 600-char chunks diluting the 1–2 relevant sentences — *not* a reranker or
threshold problem (a proposed `reranker_score_threshold=0.10` band-aid was rejected). Adopted
LangChain `SemanticChunker` (percentile breakpoint, default 95), pinning
`langchain-experimental==0.3.4` after `0.4.1` was found to require langchain-core ≥1.0.

> **From my notes:** re-ranker distribution / threshold… re-ranker scores were very low, like 0.01, etc, but upon closer inspection: sometimes, the chunk is long (1-3 paragraphs), and the relevant chunk exists in the paragraph, it is just 1-2 sentences though. So the real cause was not that we should re-tune the reranker, but instead, re-evaluate our chunking. How can we make the chunking more semantic instead of cut off by 600 words? —> semantic chunker. DANG u learn a lot by going under the hood! All about chunking, evaluation, observability, lol! Less so about embeddings.

> **From my notes:** do you see how I came to this claim upon closer inspection instead of assuming the threshold is too low? Update your system instructions so that you are able to probe deep like me across the entire architecture as to why this is happening and how we can confirm instead of just suggesting bandaid fixes.

**LlamaParse replaces pypdf (first migration).** PDFs are table/column-heavy; pypdf flattens them.
Chose LlamaParse `result_type="markdown"`. `scripts/ingest.py` gained `--reset`.

> **From my notes:** llamaindex > pypdf.

**Self-hosted reranker.** Pinecone Inference hit `RESOURCE_EXHAUSTED` (500 reranks/month). A Cohere
"SOTA" recommendation was **walked back** — bge-reranker-v2-m3 is already top-tier; the wall was
hosting, not quality. Self-hosted the *same* model via `CrossEncoder`, preserving eval
comparability, with **MPS (Metal) acceleration** (CPU 12s/3-pairs → MPS 778ms/15-pairs).

> **From my notes:** pinecone timeout issue → same model locally hosted instead bc pinecone limit on inference. This was the bge reranker model.

**Cross-vendor judge + multi-run.** After a gpt-4o 429, switched the judge to
`claude-sonnet-4-5-20250929` (`ChatAnthropic`) per the RAGAS skill's cross-vendor pitfall.
`--runs N` multi-run averaging (mean ± stddev) and `--parallel-runs` added. Reports renamed
`ragas_* → eval_*` (name artifacts by type, not toolchain). **Baseline: faithfulness 0.703–0.774,
answer_relevancy 0.770, context_precision 0.892; target ≥0.9.**

> **From my notes:** re-running 3 times to account for variability. Parallel runs, so 3 times are parallel not sequential, to save time. But capped at max_workers to not run into 429s / too many requests. Improving latency.

### Phase 3 — HyDE experiment & priority eval (2026-05-21 → 06-01)

**Index-corruption incident.** All tests suddenly `no_results`. Diagnosis (via new
`diagnose_retrieval.py` / `diagnose_index.py`): the index held **70 vectors, 100% WHO** — an
interrupted LlamaParse re-ingest, *not* the reranker (which showed sensible scores). Fix: rewrote
`ingest.py` from all-or-nothing to **per-file commit** with try/except + an `--only <source>` retry
flag, so an interruption can't wipe completed files.

**Langfuse v4.6.1 `@observe(name=)` bug.** The trace tree collapsed to `start → embedding → end`.
Real cause: `@observe(name=…)` silently drops the name in v4.6.1 (confirmed via direct API, without
the shim). Fix: rewrote the shim to call `start_as_current_observation(name=…, as_type=…)`
directly. (An interim `update_current_span(name=…)` attempt failed — "no active span" — and was
discarded.)

**The `replace_all` incident.** An eval "ran" but produced no report. A `replace_all` on
`return 0 if all(...) else 1` (which existed in two structural contexts) had **dedented an early
return out of its `if not answer_cases:` block to function scope**, making all scoring +
report-writing unreachable. Fixed by re-indenting, verified via AST. This incident is the origin of
the **auto-update-memory meta-rule (R1)** and the **replace_all per-site-verification rule (R9)**.

> **From my notes:** have claude update its system instructions every time it makes a mistake or assumption or introduces a bug to ensure it never happens again lol. To share the root cause, etc.

> **From my notes:** you've honestly made a few mistakes that I had to push back on. Go through each one and add it into your system instructions so that we backfill all the previous mistakes so they don't propagate further either. Don't make assumptions, don't hallucinate, think of downstream effects like a sharp ai engineer, walk through trade-offs, etc.

**Eval assertion redesign.** `cites_org` → `cites_org_one_of` was added and then **fully reverted**:
`[MoHFW, FOGSI, WHO]` is mathematically vacuous (those are the only sources → equivalent to
`behavior: answer`). Replaced by a universal `_check_priority_honored()` — of the sources *present*
in the retrieved chunks, assert the first cited is the highest-priority present. This tests the
priority-sort logic directly without per-case bookkeeping.

> **From my notes:** I think I care more about… if the chunks have all 3 sources, default to MoHFW, then FOGSI, then WHO / "For every answer case, check that the first cited source is the highest-priority one PRESENT in the retrieved chunks." —> made it strict. I'm sure use cases where mixed will make more sense. But clearer separation of concerns and easier prototype to manage while still showcasing my thought process. Also assuming that the precision of the chunks are good — that is measured by a different metric anyway (how good the chunks are). Then, is the ordering maintained?

**HyDE — implemented, measured, abandoned (dead-end).** HyDE v1.0 (Poshan-Saathi-voice
hypothetical) tanked scores; v2.0 (clinical-guideline voice + enforced diet filter) was still
worse. Five documented reasons it hurts *this* corpus: (1) structured guidelines don't embed near
prose hypotheticals; (2) the strong cross-encoder already bridges the query↔doc gap; (3)
text-embedding-3-small already handles conversational queries; (4) profile personalization dilutes
the embedding to a centroid; (5) retrieval surfaced chunks the LLM couldn't ground. Kept OFF, code
retained as an honest-negative portfolio result. Origin of the **HyDE-applicability rule (R21)**.

### Phase 4 — Personalization & validator (2026-06-01 → 06-03)

**Confounded regression, caught.** answer_relevancy 0.868 → 0.715 was blamed on LlamaParse and
LlamaParse was reverted to pypdf — but a git-bisect of `pipeline.py` found the real culprit was
prompt commit `f24e132` ("Improve faithfulness via tighter system prompt"), whose few-shot examples
literally opened with "The guidelines recommend…" and taught the bureaucratic register globally.
The correction (LlamaParse was never the cause; the prompt was) is the canonical
**confounded-attribution lesson (R11)**. A partial prompt revert recovered relevancy to 0.77, then
over-corrected (dropped LEAD-WITH-SUBSTANCE and the diet-filter example, which were doing real
work) and was re-added in style-safe (fluent, not bureaucratic) form.

> **From my notes:** we will have to do a diagnose on any other contradictions in our prompts —> contradicted our not-leading-with-substance guidelines with our examples.

**Schema-driven personalization.** To enforce *all* profile fields (diet, condition, trimester)
without rigidity, a `PreferenceEnum` base with `to_prompt_rule()` uses `__init_subclass__` to
enforce **at class-definition time** that every enum value has a non-empty rule (fails at import,
not at request). `UserProfile.to_personalization_block()` assembles a bulleted block injected into
the **user** message (keeping the system prompt static for prompt-cache hits). faithfulness → 0.89.
11 parametrized tests.

> **From my notes:** but what if I add different profile inputs in the future? This would make it rigid / inflexible? —> not listening to user profile… how to personalize.. but not make it rigid. Structure / schema enforced.

**Post-answer validator.** A hardcoded forbidden-terms list was **rejected** (too rigid, misses
regional foods, can't tell "avoid chicken" from "include chicken"). Built instead as an
**LLM detect-and-fix** in one call (`validator.py`, `gpt-4.1-nano` temp 0, structured output),
regex-gated so it short-circuits when a profile has no restrictions. Validates three dimensions:
DietType exclusion, hypertension (sodium), diabetes (refined-carb). A `"{"`-in-answer bug (the LLM
was asked to echo the compliant answer and returned broken JSON) taught **R12: implement the no-op
branch in code, not by asking the LLM to echo.**

> **From my notes:** validation to enforce diet types… still get back meat for ovo vegetarian, etc. Validator that checks. And if found that it invalidates, then it removes those fields, does not add new info.

**Reasonable-inference rules + LlamaParse re-migration.** ALLOWED/FORBIDDEN inference rules were
added to the system prompt (a recommended food → safe; "such as" → examples; but no
topic-substitution, no content-filling, no supplement↔food switching). LlamaParse was then
**re-migrated deliberately** ("we learned the drop was the prompt, not LlamaParse; do NOT change the
prompt, just migrate") with a measured A/B: context_precision 0.828 → 0.882, faithfulness
0.793 → 0.710. Chunking overhaul: breakpoint 95 → 85, `chunk_max_tokens=512` backstop.

> **From my notes:** given we are using semantic chunking, the chunks are not fixed length. How does that affect any type of token limit, in either models or rerankers or anywhere else that may truncate at 512 tokens for instance?

### Phase 5 — Hybrid search & header chunking (2026-06-03 → 06-15)

**Multi-run Langfuse fixes.** Three bugs: scores attached only to run 1 (origin of **R29** — verify
side effects apply to *all* N iterations); trace-ID contamination (RAGAS `evaluate()` sets
contextvars that persist, so `get_current_trace_id()` returned stale RAGAS IDs → wrapped each
`run_chat()` in `contextvars.Context().run()`); and a NaN score (from a mid-run credit failure)
that passed the `is None` check but was rejected by Langfuse → `math.isnan` guard.

**Hybrid BM25 + dense.** `pinecone-text` BM25 encoder, `hybrid_alpha=0.75` (client-side scaling,
since the classic Pinecone API has no server-side alpha), index metric changed cosine → dotproduct,
2-pass ingest (fit IDF on the full corpus first). Intent: surface rare Indian-context keywords.
Result: context_precision **dropped to 0.688** — the `0.3` threshold is calibrated for cosine and
mis-fires on the compressed hybrid range, and "water" has near-zero IDF. This motivated the
threshold recalibration work.

> **From my notes:** hybrid search. Improved answer relevancy but decreased context_precision. Go back to first part of pipeline: quality of chunks. Reinvest with headers separated after seeing answer outputs with too many different ideas in the chunks.

> **From my notes:** interdependency —> adding hybrid search meant that for water queries, it would be filtered out so had to lower the similarity_threshold from 0.1 to 0.05 to account for it, while still being able to retrieve relevant chunks for "amla" etc — what a delicate balance / dance!

> **From my notes:** next up: tuning semantic chunking — more sensitivity and also smaller chunk sizes so not so irrelevant.

**Header-aware chunking.** `MarkdownHeaderTextSplitter` (#/##/###) + a breadcrumb section prefix,
in a 4-stage pipeline (header split → prefix → SemanticChunker within section → token cap). Fixed
mixed-content mega-chunks (calcium + albendazole + thyroid in one chunk). Chunk count 494 → **903**.

**Two discipline moments.** A header+MMR+threshold bundle was started *without permission* → user
"STOP" → **reverted** (hybrid work kept); reinforced one-change-at-a-time (**R4/R5**). And a
proposal to set `similarity_threshold=0.0` was **rejected hard** ("NO filtering at the first
layer") — the lesson being recalibrate, don't remove. **MMR was planned repeatedly but never
implemented.**

> **From my notes:** restructured memory.md when it got too big at 1K lines: an index of `## ACTIVE RULES — CHECK BEFORE ACTING | # | Rule | Trigger |`.

**Prompt hardening (dead-end) + profile-aware reranker.** Track 1 (prompt v1.2 — causal-wrap-up and
cross-purpose-mapping prohibitions) produced *no measurable lift* — a dead-end. Track 2 replaced the
terse `[Diet: X]` reranker tag with a natural-language `_build_reranker_query()` prose string
(e.g. "vegetarian (no meat/poultry/fish/eggs) pregnant woman, second trimester: iron sources"), so
the cross-encoder sees profile context in the form it was trained on.

### Phase 6 — Faithfulness endgame (2026-06-15 → 06-30)

**Two-call answer post-processing.** `review_answer` (gpt-4.1-mini, RAGAS-style atomic-claim
decomposition — verify each claim against context, drop the unsupported) + `validate_and_fix`
(diet/safety + deflection rewrites). The judge model defaults to gpt-4.1-mini, never weaker than
the answer model (detecting an ungrounded claim is at least as hard as generating one).

**R31 — deterministic enforcement over soft rules.** gpt-4.1-mini/nano bypass prose prompt rules
even at temperature 0 (confirmed repeatedly — an enumeration bullet was inert; a condensed prompt
still emitted a benefit-closer). The resolution: move every *detectable* defect into deterministic
code — regex strippers for deflective openers, embedded hedges, cadence/frequency riders
("for at least 180 days" must never become "daily"), and a deterministic answerability gate (a
quantity question answered with no quantity routes to `no_results`).

> **From my notes:** ragas validator strict… even if answer seems relevant, if it starts with a more deflective answer (like "not found in context, but smth similar in context is this…") it gives it 0. So added regex to lead with substance to go around ragas validator to not flag them as 0. In future could tweak or design your own validation.

> **From my notes:** scoring evasive with ragas, so put substantial in the front so not triggering.

**Answerability reclassifications.** `water_intake` and (transiently) `paneer_intake` were routed to
`no_results` where the corpus has no grounded answer; `folic_acid` was pivoted from the unanswerable
"which foods" to the answerable "how much." `paneer_intake` was later flipped *back* to an answer
case once retrieval reliably surfaced the plate-method portion guidance (the corpus's real answer to
"how much").

**The variance diagnosis and temperature 0.** Single-run faithfulness is variance-dominated: the
same *byte-identical* temp-0 answer scores 0.0–0.78 across judge runs (a pure judge-side floor,
unfixable via prompts). The earlier 2026-05 "temp 0.1 is worse" conclusion was itself a single-run
measurement of noise. Setting **`llm_temperature` 0.3 → 0** fixes the answer text across runs,
isolating grounding logic from sampler noise; grounding stays enforced by the review pass +
strippers. The answer model default moved **nano → gpt-4.1-mini** (identical faithfulness, a
reliable answer_relevancy gain 0.88 → 0.93).

> **From my notes:** between nano and mini models, nano is cheaper faster whereas mini is more complex. Despite this, for our objectives, nano performs just as well as mini, so no need to upgrade tbh! The fast models are great at more classifying / summarizing / easy tasks! It's awesome! Little overhead. The canonical number is a multi-run average, guarded
by a disk watchdog (`_run3_watchdog.sh`) because the reranker + RAGAS scratch can OOM-SIGKILL on a
full disk.

**Prompt condensing (v1.5 → v1.6).** The answer system prompt was condensed ~50 → ~30 lines
(grounding + fewest-claims made salient, enumeration discipline folded in, examples cut to one
generic — removing a test-data-in-examples R28 violation). Score-neutral (0.853 → 0.844, within
noise), regression-free, and strictly better as an artifact.

> **From my notes:** I like the proposal BUT baking in the answer to one of the test cases feels cheating. Propose a different way, maybe using a different name instead of folic acid or whatever you think is SOTA.

**Final measured scores** (run 20260630_183510, temp-0 verify): **faithfulness 0.853 /
answer_relevancy 0.933 / context_precision 0.917** — the first run to clear the ≥0.85 faithfulness
bar with the other two comfortably above 0.90.

> **From my notes:** has been really fun looking at actual outputs and seeing where to make tweaks… in a smaller subset can look at it manually, but larger case, look at the eval scores, see which are low and look into them… what is the answer? Non-committal? What do the chunks look like? Update on reranking / selection of chunks? Update on prompt? Chunks have answers but the prompt is too strict? Loving this lol.

---

## Part III — Measured score timeline

Every RAGAS run in `eval/results/`, with its tuning note. `F` = faithfulness, `R` = answer_relevancy,
`P` = context_precision (mean where the run averaged ≥2 iterations). Runs before 05-06 predate
RAGAS scoring (routing-only). This is the empirical spine of Parts I–II.

> **From my notes:** the shorthand I kept jotting per run — F / R / P, e.g. `0.76 0.81 0.75`.

| Date · time | F | R | P | What changed |
|---|---|---|---|---|
| 0428_1358–1402 | – | – | – | similarity_threshold sweep 0.4/0.5/0.3/0.4/0.2/0.3 |
| 0428_1424 | – | – | – | baseline after schema-first audit |
| 0428_1452 | – | – | – | regex guardrails → LLM classifier |
| 0428_1853 | – | – | – | baseline after Langfuse v4 wiring |
| 0506_1221 | – | – | – | semantic chunking (was fixed) |
| 0506_1300 | 0.842 | 0.759 | 0.725 | judge → gpt-4o (stronger) |
| 0506_1325 | 0.703 | 0.761 | 0.892 | judge → cross-vendor Claude (anti-inflation) |
| 0506_1332 | 0.774 | 0.770 | 0.892 | re-run for variance (Claude judge) |
| 0506_1346 | 0.704 | 0.768 | 0.892 | answer LLM → gpt-4.1-mini |
| 0508_1747 | 0.625 | 0.820 | 0.843 | prompt refine to reduce hallucination |
| 0508_1757 | 0.715 | 0.815 | 0.824 | re-run for variability |
| 0508_1822 | 0.802 | 0.870 | 0.843 | 3-run avg; "lead with substance" prompt |
| 0508_1834 | 0.797 | 0.868 | 0.876 | reranker_candidate_k 5 → 3 |
| 0515_1137 | – | – | – | post-LlamaParse re-ingest (first migration) |
| 0521_1706 | – | – | – | LlamaParse + self-hosted bge reranker |
| 0523_1521 | 0.804 | 0.756 | 0.865 | re-run after indentation fix; parallel-runs |
| 0523_1638 | 0.825 | 0.718 | 0.835 | quick re-run (Langfuse trace check) |
| 0526_1757 | 0.850 | 0.696 | 0.843 | priority-honored eval layer |
| 0527_1354 | 0.769 | 0.618 | 0.795 | **HyDE v1.0 on** |
| 0601_1508 | 0.770 | 0.516 | 0.799 | **HyDE v2.0** (clinical voice + profile) |
| 0601_1544 | 0.843 | 0.715 | 0.853 | **HyDE off** — baseline restored |
| 0601_1622 | 0.873 | 0.757 | 0.877 | remove LlamaParse → pypdf (confounded revert) |
| 0601_1637 | 0.785 | 0.771 | 0.863 | revert to less-general, warmer prompt |
| 0601_1947 | 0.899 | 0.762 | 0.877 | **schema-driven personalization** |
| 0602_1451 | 0.880 | 0.805 | 0.833 | diet/BP/diabetes **validator** added |
| 0602_1528 | 0.758 | 0.769 | 0.828 | fix `{` bug + deflective-opener check |
| 0602_1543 | 0.817 | 0.814 | 0.792 | top_k 3 → 5 |
| 0602_1606 | 0.765 | 0.813 | 0.750 | prompt less strict; trailing-deflection check |
| 0602_1631 | 0.793 | 0.873 | 0.828 | reasonable-inference + supplement pivot |
| 0602_1639 | 0.797 | 0.868 | 0.767 | test top_k = 5 |
| 0602_1909 | 0.710 | 0.878 | 0.882 | **LlamaParse re-migration** (top_k 3) |
| 0602_1929 | 0.798 | 0.866 | 0.818 | top_k 3 → 5 on LlamaParse |
| 0602_1943 | 0.832 | 0.926 | 0.843 | tighten prompt: no invented benefits |
| 0602_1950 | 0.817 | 0.872 | 0.757 | top_k 3 → 5 |
| 0602_2004 | 0.733 | 0.923 | 0.765 | no causal A→B transfer (top_k 3) |
| 0602_2047 | 0.713 | 0.918 | 0.804 | re-ingest breakpoint 0.95 → 0.85 |
| 0602_2059 | 0.794 | 0.865 | 0.750 | top_k 3 → 5 post-rechunk |
| 0602_2119 | 0.855 | 0.790 | 0.784 | **consolidate grounding/forbidden rules** |
| 0602_2126 | 0.842 | 0.896 | 0.794 | (that consolidation, top_k 5 → 3) |
| 0602_2133 | 0.779 | 0.904 | 0.810 | 3 runs at top_k 5 (variance) |
| 0603_1426 | 0.827 | 0.856 | 0.840 | 3 runs top_k 3 (Langfuse multi-run check) |
| 0603_1507 | 0.813 | 0.883 | 0.763 | 3 runs candidate_k 5 / top_k 3 |
| 0604_1813 | 0.806 | 0.880 | 0.521 | 3 runs candidate_k 5 / top_k 5 |
| 0605_1519 | 0.844 | 0.907 | 0.688 | **hybrid BM25 + dense** (alpha 0.75) |
| 0608_1401 | 0.807 | 0.855 | 0.804 | similarity_threshold → 0.1 |
| 0608_1418 | 0.806 | 0.898 | 0.706 | threshold → 0.5 |
| 0608_1427 | 0.828 | 0.903 | 0.686 | threshold → 0.3 baseline |
| 0608_1439 | 0.730 | 0.858 | 0.760 | threshold → 0.4 |
| 0608_1502 | 0.726 | 0.895 | 0.642 | **header-split chunking** (903 chunks) |
| 0608_1529 | 0.670 | 0.866 | 0.672 | fix contradictory folic-acid examples |
| 0608_1537 | 0.633 | 0.865 | 0.626 | candidate_k 5 / top_k 5 (small chunks) |
| 0608_1602 | 0.710 | 0.921 | 0.636 | threshold 0.05, candidate_k 5, top_k 5 |
| 0608_1610 | 0.775 | 0.861 | 0.755 | threshold 0.05, candidate_k 5, top_k 3 |
| 0608_1619 | 0.715 | 0.918 | 0.745 | (corrected: 0.05 / 5 / 3) |
| 0615_1404 | 0.632 | 0.925 | 0.735 | prompt v1.2 hallucination rules (dead-end) |
| 0615_1421 | 0.758 | 0.857 | 0.853 | **profile-aware reranker query** |
| 0615_1440 | 0.705 | 0.917 | 0.819 | llm_temperature 0.3 → 0.1 |
| 0615_1521 | 0.728 | 0.853 | 0.843 | **grounding validator** (atomic claims) |
| 0615_1613 | 0.825 | 0.791 | 0.906 | candidate_k 5 → 3 (drop weak chunks) |
| 0615_1646 | 0.853 | 0.842 | 0.896 | validator hardening + no_results routing |
| 0624_1544 | 0.838 | 0.902 | 0.833 | paneer → no_results |
| 0624_1643 | 0.808 | 0.927 | 0.956 | water → no_results; review_answer folds grounding+answerability |
| 0625_2028 | 0.833 | 0.900 | 0.926 | (re-run) |
| 0625_2144 | 0.801 | 0.906 | 0.833 | prompt v1.3 (quantitative dosages, meal assignment) |
| 0626_1029 | 0.811 | 0.892 | 0.922 | v1.3 without over-detailing (fewer no_results) |
| 0627_1458 | 0.812 | 0.910 | 0.800 | v1.3 + review v2.1 grounding/answerability |
| 0627_1525 | 0.820 | 0.907 | 0.828 | review v3.0 — deterministic quantity gate |
| 0627_1555 | 0.805 | 0.836 | 0.922 | water = no_results; atomic decomposition |
| 0627_1706 | 0.869 | 0.906 | 0.889 | (per-case faithfulness matrix run) |
| 0627_1722 | 0.779 | 0.896 | 0.956 | (variance run) |
| 0627_1731 | 0.767 | 0.931 | 0.927 | (variance run) |
| 0627_1736 | 0.853 | 0.933 | 0.903 | (variance run — nano/mini A/B) |
| 0630_1835 | 0.853 | 0.933 | 0.917 | **temp 0** verify (mini, validator v2.4) |
| 0630_1939 | 0.844 | 0.934 | 0.844 | **v1.6 condensed prompt** (score-neutral) |

The timeline reads as a plateau, not a climb, and that is the honest finding: after ~0508 the
three metrics oscillate inside a variance band (roughly F 0.70–0.87, R 0.75–0.93, P 0.65–0.96)
that is wider than most single-change deltas. The durable gains came from a few structural moves
(cross-vendor judge, schema personalization, the review/validator pass, candidate_k 5 → 3 for
precision) and from *reducing variance* (temperature 0, multi-run averaging) — not from any single
prompt tweak.

---

## Part IV — Dead-ends, reversals, and things deliberately not built

Kept explicitly, because the measured negatives are part of the engineering record.

**Reverted after measuring:**
- **HyDE** (v1.0 and v2.0) — degraded every metric on a structured-guideline corpus; kept OFF.
- **First LlamaParse migration** — reverted to pypdf on a quality drop, then proven a confounded
  variable (the real cause was a prompt commit) and re-migrated deliberately.
- **`cites_org_one_of` eval assertion** — added, found mathematically vacuous, fully reverted for
  the universal priority-honored check.
- **Header + MMR + threshold bundle** — reverted for being started without permission (hybrid work
  within it was kept).
- **`llm_temperature = 0.1`** — the "0.1 is worse" conclusion was a single-run measurement of noise;
  superseded by temperature 0 for reproducibility.
- **Prompt v1.2 hallucination rules** — no measurable lift.

**Rejected before building:**
- **Custom RAGAS-style judge** — reversed in favor of the RAGAS library.
- **Hardcoded forbidden-terms validator** — too rigid; became the LLM detect-and-fix validator.
- **`reranker_score_threshold=0.10`** and **`similarity_threshold=0.0`** — band-aids; the real fixes
  were better chunking and threshold recalibration respectively.
- **Cohere reranker** — "SOTA" framing walked back; self-hosted the same bge model instead.
- **Additive source-priority score nudges** — untunable hyperparameters; preference lives in
  context ordering.
- **Langfuse Prompt Management** — would make Langfuse a hard runtime dependency; kept code-versioned
  `PROMPT_VERSION` constants instead.

**Planned but never implemented:**
- **MMR** (repeatedly discussed for diversity/redundancy).
- **Contextual Retrieval** (Anthropic 2024 — per-chunk LLM blurb before embed/BM25).
- **Table-aware chunking** (keep a table intro bound to its markdown table).
- **Guaranteed source floor** (superseded by the reranker/MMR discussion).
- **Health endpoint** (`/health/index`, startup + eval pre-flight).
- **Frontend** (Next.js) and **deployment** (Railway/Fly backend + Vercel frontend).

---

## Part V — Commit index (git backbone)

54 commits on `main`. The load-bearing ones, in order:

- `e781e88` — schema-first audit (typed module boundaries, ResponseType enum)
- `1119721` — LLM classifier replaces keyword guardrails
- `e94581b` — Langfuse v4 observability
- `9cc6aa6` — RAGAS eval + reusable RAGAS/Langfuse skills
- `131c008` — semantic chunking + combined routing/RAGAS report + cross-vendor judge
- `eaf0515` — reranker `max_length=8192` pin + assertion
- `f24e132` — "tighter system prompt" (later identified as the bureaucratic-register regression)
- `c3b3dc6` — schema-driven personalization + validator + pypdf revert
- `0eb86a0` — REASONABLE INFERENCE ALLOWED/FORBIDDEN rules (R2 corrected)
- `444e979` — LlamaParse re-migration + `llama-cloud==0.1.46` pin + measured A/B
- `91bbd04` — Langfuse multi-run score fixes (R29, contextvars isolation, NaN guard)
- (header-aware chunking commit — 903 chunks)
- `275b1b0` — deterministic answerability gate
- `3f2aad5` — safety-hedge catch + duration→frequency curb
- `b7b632d` — deterministic frequency-claim stripper
- `086403f` — default answer model → gpt-4.1-mini
- `be02701` — answer temperature → 0 for reproducible faithfulness
- `feb6495` — disk-guarded 3× eval sweep helper
- `e5b534b` — paneer_intake → answer case
- `3eab2c6` — condense answer system prompt (v1.5 → v1.6)

---

*Sources: full development-chat transcript (six mined time-slices, 2026-04-09 → 06-30); `git log`
(54 commits); 92 RAGAS reports in `eval/results/`. Assembled 2026-06-30.*
