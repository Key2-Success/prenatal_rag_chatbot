"""
validator.py — Post-answer validator with auto-correction.

Two classes of violation are caught here:

  1. Dietary restrictions (from the user's profile via to_validation_rule).
     Soft prompt enforcement leaks on diet ~5-10% of the time; this is the
     hard-enforcement backstop. Catches non-veg foods recommended to a
     vegetarian, etc.

  2. Deflective openers (universal, profile-independent). Phrases like
     "The guidelines do not specify..." trigger RAGAS's noncommittal
     classifier and score answer_relevancy to 0 even when the rest of the
     answer is substantive. Detected by a regex pre-check on the answer's
     opening (cheap, deterministic) — the LLM only rewrites it when
     detected. The regex isn't trying to be exhaustive; it's the trigger
     for invoking the LLM rewriter, which can handle nuance.

Cost gating (the user explicitly asked for this — see MEMORY.md):
  - No profile rules AND no deflective opener detected → 0 LLM calls. Done.
  - Otherwise → 1 LLM call that handles BOTH dimensions in one pass.

No-op safety (also see MEMORY.md):
  When the validator's classification step says is_compliant=True, the
  CODE (not the LLM) returns the original answer unchanged. We don't trust
  the LLM to echo ~1KB of text verbatim through structured output — that's
  how an earlier version of this module returned `corrected_answer = "{"`
  for a perfectly good answer.
"""

import re

from pydantic import BaseModel, Field

from backend.app.clients import get_openai_client
from backend.app.config import settings
from backend.app.models.schemas import UserProfile
from backend.app.observability import observe, update_current_span

PROMPT_VERSION = "v2.2"  # v2.2: added embedded quantitative-hedge detection + rule

# Versioned independently of PROMPT_VERSION above — review_answer is a
# separate LLM call with its own prompt, so its version moves on its own.
# v2.0: merged answerability judgment into the faithfulness pass (one call).
# v2.1: two complementary answerability refinements (subject-pivot override +
#       what/which-foods scope note).
# v3.0: REMOVED answerability from this LLM call entirely. The soft-prompt
#       answerability verdict proved non-deterministic at temperature 0 (the
#       same case oscillated TRUE/FALSE across runs; layering more prose rules
#       in v2.1 did not fix it). Per the project's "soft rules that bypass →
#       enforce deterministically" pattern, answerability is now a narrow
#       regex gate (check_answerability, below) wired into the pipeline. This
#       call is back to PURE faithfulness — one job, judged reliably.
# v3.1: tightened the METHOD decomposition step (not a new rule — a granularity
#       fix). The "added purpose/benefit" rule already existed but did not fire
#       because the judge kept a grounded fact and its tacked-on ungrounded
#       purpose ("…to support health and development") in ONE coarse claim, and
#       the grounded half shielded the bad half. RAGAS decomposes atomically and
#       failed it; our judge under-split and passed it. Step 1 now forces
#       purpose/benefit clauses and per-item attributions into separate claims
#       so the existing rule can actually catch them.
REVIEW_PROMPT_VERSION = "v3.1"

# Forbidden opener patterns. Regex used ONLY to decide whether to invoke
# the LLM rewriter — NOT to classify or filter content (where LLM judgement
# is needed). These patterns are literal English idioms the answer LLM
# tends to overuse; they're small, well-defined, and stable.
#
# Case-insensitive, matched against the first ~120 chars of the trimmed
# answer. Includes "the guidelines do not", "the context does not",
# "I don't have", and common variants.
_FORBIDDEN_OPENER_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in (
        r"^\s*the guidelines (?:do not|don'?t)\b",
        r"^\s*the context (?:does not|doesn'?t)\b",
        r"^\s*the (?:provided )?(?:documents?|text|guidelines?) (?:does not|doesn'?t|do not|don'?t)\b",
        r"^\s*there (?:is|are) no\b",
        r"^\s*i (?:don'?t|do not) have\b",
        r"^\s*guidelines (?:do not|don'?t) specify\b",
        r"^\s*(?:unfortunately|while|although)[, ]+the (?:guidelines|context|documents?)\b",
    )
]

_OPENER_RULE = (
    "OPENER REWRITE — REQUIRED. An automated regex pre-check has confirmed "
    "that the answer starts with a forbidden deflective phrase such as "
    "'The guidelines do not...', 'The context does not...', 'There is no...', "
    "'I don't have...', 'Unfortunately the guidelines...', or similar. "
    "This IS a confirmed violation — you do not need to re-verify it. "
    "Your job: set is_compliant=false, add a violation with field='opener', "
    "and produce a rewritten corrected_answer that:\n"
    "  * Leads with the substantive guidance the original answer already "
    "contains (do NOT add new facts).\n"
    "  * REMOVES the deflective phrasing entirely — do NOT just move it to "
    "the end. If the body of the answer is complete and self-contained, "
    "drop the caveat. Only KEEP a brief 'the guidelines don't list an "
    "exact X per Y' note at the end if it genuinely adds value the answer "
    "doesn't already convey.\n"
    "  * Preserves the original voice, tone, length, and all factual "
    "content other than the deflective opener.\n"
    "Example — original: 'The guidelines do not specify an exact amount of "
    "paneer. However, include protein-rich foods like paneer.' → rewritten: "
    "'Include protein-rich foods like paneer in your meals.' (drop the "
    "caveat — the answer is complete)."
)

_TRAILING_RULE = (
    "TRAILING DEFLECTION REWRITE — REQUIRED. An automated regex pre-check "
    "has confirmed that the LAST sentence of the answer is a deflective "
    "phrase like 'The guidelines do not specify an exact number of...' or "
    "'The guidelines don't list...' that adds no useful content. The body "
    "of the answer already gives a satisfactory response; the trailing "
    "sentence is noise. "
    "Your job: set is_compliant=false, add a violation with field='trailing', "
    "and produce a corrected_answer that REMOVES the trailing deflective "
    "sentence entirely. Keep the rest of the answer UNCHANGED — do not "
    "rewrite, paraphrase, or add anything to the substantive body.\n"
    "Example — original: 'Drink 2-3 liters of water per day. The guidelines "
    "do not specify an exact number of glasses.' → rewritten: 'Drink 2-3 "
    "liters of water per day.' (just drop the last sentence)."
)

# Embedded quantitative hedges. Unlike opener/trailing (which look only at the
# first/last sentence), these can sit ANYWHERE in the answer body — e.g.
# "...but do not specify the exact amount for daily consumption". RAGAS's
# noncommittal classifier scores answer_relevancy to 0 on this phrasing
# wherever it appears, even surrounded by substantive guidance. Narrow on
# purpose (only the "no exact/specific amount" family) so we don't fire on
# legitimate mid-answer "there is no..." phrasings.
_EMBEDDED_HEDGE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in (
        r"do(?:es)?\s+not\s+specify\s+(?:an?\s+|the\s+)?(?:exact|specific|precise)\b",
        r"don'?t\s+specify\s+(?:an?\s+|the\s+)?(?:exact|specific|precise)\b",
        r"\bno\s+(?:exact|specific|precise)\s+(?:amount|number|quantity|dose|dosage|serving)\b",
    )
]

_EMBEDDED_HEDGE_RULE = (
    "EMBEDDED HEDGE REMOVAL — REQUIRED. An automated regex pre-check has "
    "confirmed the answer contains a quantitative deflection such as "
    "'...do not specify the exact amount' or '...does not specify a precise "
    "number' embedded in the body — not just at the start or end. RAGAS's "
    "noncommittal classifier scores this as evasive even when the surrounding "
    "answer is substantive. This IS a confirmed violation; you do not need to "
    "re-verify it. Your job: set is_compliant=false, add a violation with "
    "field='embedded_hedge', and produce a corrected_answer that REMOVES only "
    "the hedge clause while keeping the recommendation intact. Do NOT add new "
    "facts.\n"
    "Example — original: 'Include a variety of whole grains in your meals, but "
    "the guidelines do not specify the exact number of servings per day. "
    "Choose options you enjoy.' → rewritten: 'Include a variety of whole "
    "grains in your meals. Choose options you enjoy.' (drop only the hedge "
    "clause; keep the recommendation)."
)


class _Violation(BaseModel):
    """One detected violation of a profile-driven or universal rule."""
    field: str = Field(
        ...,
        description="Which rule was violated (e.g. 'diet_type', 'hypertension', 'opener').",
    )
    violating_foods: list[str] = Field(
        default_factory=list,
        description="The exact phrases from the original answer that violated the rule. May be empty for opener violations where the violation is the SHAPE of the opening rather than specific foods.",
    )
    explanation: str = Field(
        ...,
        description="One-sentence reason this is a violation.",
    )


class ValidationResult(BaseModel):
    """
    Output of validate_and_fix.

    is_compliant: True if the ORIGINAL answer had no violations.
    violations: list of detected violations (empty when is_compliant).
    corrected_answer: the answer text to actually use downstream.

      IMPORTANT: when is_compliant=True, this field is set to the ORIGINAL
      answer text by the calling code, NOT by the LLM. The LLM's job is
      classification + (when needed) rewriting; the code handles the echo.
      See MEMORY.md "Don't trust LLMs to ECHO content" for why.
    """
    is_compliant: bool
    violations: list[_Violation]
    corrected_answer: str


_SYSTEM_PROMPT = """You check whether an LLM-generated answer complies with a set of rules, AND produce a corrected version when violations are found.

You will receive:
  1. One or more RULES — each describes either a dietary restriction or a universal style rule, and what to flag.
  2. An ANSWER to validate.

For each rule, identify any way the answer violates it. Do NOT flag mentions where the answer is telling the user to AVOID restricted foods — that is the answer correctly enforcing the rule.

Then produce the `corrected_answer` field:

  - If you found violations, produce a corrected version. Constraints:
      * For DIETARY violations: REMOVE the violating foods. Do NOT add new foods, new claims, or new facts. Preserve voice, tone, sentence structure, and length as closely as possible. Smooth partial sentences minimally (e.g. "include eggs, paneer, and dal." instead of "include eggs, paneer, dal, and chicken.").
      * For OPENER violations: REWRITE the opening to lead with the substantive guidance the answer already contains. Move any deflective phrasing to the END of the answer, briefly, if at all. Do NOT add new facts or invent new specifics — only re-order and rephrase what's already in the answer.
      * If removal would leave nothing meaningful, use a brief fallback like "Good options from the guidelines include the plant-based foods listed."

  - If you found NO violations (the answer is fully compliant), set corrected_answer to the EMPTY STRING "". The calling code will use the original answer instead. Do NOT attempt to echo or paraphrase the original — leave corrected_answer empty when compliant.

Output structured JSON with is_compliant (bool), violations (list, empty when compliant), and corrected_answer (rewritten text when there are violations, empty string when compliant).
"""


def _collect_dietary_rules(profile: UserProfile) -> list[str]:
    """Walk the profile's rule providers and collect every non-None to_validation_rule()."""
    rules: list[str] = []
    for provider in profile._collect_rule_providers():
        if hasattr(provider, "to_validation_rule"):
            rule = provider.to_validation_rule()
            if rule:
                rules.append(rule)
    return rules


def _has_deflective_opener(answer: str) -> bool:
    """
    Cheap regex pre-check: does the answer start with a known deflective
    phrase pattern? Used as a GATE for invoking the LLM rewriter, NOT as
    the rewriter itself. False negatives (regex misses a deflective opener
    we didn't list) are acceptable; the answer leaks through but no
    correctness harm. False positives (regex flags a non-deflective opener)
    are rarer and trigger an unnecessary LLM call — also acceptable.
    """
    if not answer:
        return False
    head = answer.strip()[:120]
    return any(pat.search(head) for pat in _FORBIDDEN_OPENER_PATTERNS)


def _has_trailing_deflection(answer: str) -> bool:
    """
    Cheap regex pre-check: does the LAST sentence of the answer match a
    known deflective phrase pattern? Symptom: the LLM gives a complete
    answer (e.g. "Drink 2-3 liters of water per day.") then trails off
    with a useless hedge ("The guidelines do not specify an exact number
    of glasses."). Same regex patterns as the opener check, just applied
    to the final sentence.

    Edge cases:
      - Single-sentence answers where the only sentence is a forbidden
        phrase: caught by the opener check, not this one. We do not
        double-flag — the opener check fires first in validate_and_fix.
      - Answers with no period (model didn't terminate): the whole
        string is treated as the last sentence; pattern may or may not
        fire depending on its start.
    """
    if not answer:
        return False
    # Split on sentence enders. Keep only non-empty after stripping.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if len(sentences) < 2:
        # Single-sentence answer — opener check handles it; don't
        # double-detect (would create spurious "trailing" violation on
        # a single deflective sentence).
        return False
    last = sentences[-1][:120]
    return any(pat.search(last) for pat in _FORBIDDEN_OPENER_PATTERNS)


def _has_embedded_hedge(answer: str) -> bool:
    """
    Detect a quantitative deflection anywhere in the answer body — e.g.
    "...but do not specify the exact amount for daily consumption". Scans the
    WHOLE answer (not just first/last sentence) because RAGAS's noncommittal
    classifier penalises this hedge wherever it sits, even mid-sentence
    between substantive clauses. Used as the GATE for invoking the LLM
    rewriter, same as the opener/trailing checks.
    """
    if not answer:
        return False
    return any(pat.search(answer) for pat in _EMBEDDED_HEDGE_PATTERNS)


@observe(name="validate_and_fix")
def validate_and_fix(answer: str, profile: UserProfile) -> ValidationResult:
    """
    Validate an answer against profile rules + universal opener rule, and
    return a (possibly corrected) version.

    Decision tree:
      1. No profile rules AND no deflective opener detected → short-circuit
         with zero LLM calls. The original answer is returned unchanged.
      2. Otherwise → one LLM call that handles all detected violations.
         When is_compliant=True comes back, CODE returns the original
         answer (no LLM-echo trust).
    """
    dietary_rules = _collect_dietary_rules(profile)
    has_opener_issue = _has_deflective_opener(answer)
    has_trailing_issue = _has_trailing_deflection(answer)
    has_embedded_hedge = _has_embedded_hedge(answer)

    if (not dietary_rules and not has_opener_issue
            and not has_trailing_issue and not has_embedded_hedge):
        update_current_span(
            metadata={"validation_skipped": "no_applicable_rules_and_clean"},
            output={"is_compliant": True, "corrected": False},
        )
        return ValidationResult(
            is_compliant=True, violations=[], corrected_answer=answer,
        )

    # Build the rules list for the LLM. Opener and trailing rules appear
    # first (most visible). Dietary rules follow. The LLM's single call
    # handles all detected dimensions in one rewrite.
    rules: list[str] = []
    if has_opener_issue:
        rules.append(_OPENER_RULE)
    if has_trailing_issue:
        rules.append(_TRAILING_RULE)
    if has_embedded_hedge:
        rules.append(_EMBEDDED_HEDGE_RULE)
    rules.extend(dietary_rules)

    update_current_span(input={
        "n_rules": len(rules),
        "has_opener_issue_pre_check": has_opener_issue,
        "has_trailing_issue_pre_check": has_trailing_issue,
        "has_embedded_hedge_pre_check": has_embedded_hedge,
        "n_dietary_rules": len(dietary_rules),
        "original_answer": answer,
        "prompt_version": PROMPT_VERSION,
    })

    response = get_openai_client().beta.chat.completions.parse(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": (
                "RULES:\n" + "\n".join(f"- {r}" for r in rules)
                + "\n\nANSWER:\n" + answer
            )},
        ],
        response_format=ValidationResult,
    )
    raw = response.choices[0].message.parsed
    if raw is None:
        raise RuntimeError(
            "Validator returned None — structured output parse failed."
        )

    # REGEX-OVERRIDES-LLM safety: if our deterministic pre-checks detected
    # opener / trailing deflection, the LLM's classification is overridden
    # — those detections are FACT, not opinion. Add the violation if the
    # LLM missed it. We don't re-call the LLM (cost / latency); we trust
    # the corrected_answer if present (the LLM produced something) or fall
    # back to the original.
    is_compliant = raw.is_compliant
    violations = list(raw.violations)
    if has_opener_issue and not any(v.field == "opener" for v in violations):
        is_compliant = False
        violations.append(_Violation(
            field="opener",
            violating_foods=[],
            explanation=(
                "Regex pre-check detected a forbidden deflective opener "
                "(LLM missed or ignored the rule)."
            ),
        ))
    if has_trailing_issue and not any(v.field == "trailing" for v in violations):
        is_compliant = False
        violations.append(_Violation(
            field="trailing",
            violating_foods=[],
            explanation=(
                "Regex pre-check detected a trailing deflective sentence "
                "(LLM missed or ignored the rule)."
            ),
        ))
    if has_embedded_hedge and not any(v.field == "embedded_hedge" for v in violations):
        is_compliant = False
        violations.append(_Violation(
            field="embedded_hedge",
            violating_foods=[],
            explanation=(
                "Regex pre-check detected an embedded quantitative hedge "
                "(LLM missed or ignored the rule)."
            ),
        ))

    # CRITICAL no-op safety: when ULTIMATELY compliant (after override),
    # CODE returns the original answer. We don't trust the LLM to echo
    # ~1KB of text through structured output. See MEMORY.md
    # "Don't trust LLMs to ECHO content."
    if is_compliant:
        final_answer = answer
    else:
        # Use the LLM's rewritten version when available; fall back to
        # the original if it's empty/whitespace (defensive — shouldn't
        # happen with the new prompt but guard against it).
        rewritten = raw.corrected_answer.strip()
        final_answer = rewritten or answer

    result = ValidationResult(
        is_compliant=is_compliant,
        violations=violations,
        corrected_answer=final_answer,
    )

    update_current_span(
        output={
            "is_compliant": result.is_compliant,
            "n_violations": len(result.violations),
            "corrected_answer": result.corrected_answer,
        },
        metadata={
            "violations": [v.model_dump() for v in result.violations],
            "answer_changed": result.corrected_answer != answer,
        } if not result.is_compliant else None,
    )
    return result


# --- Answer review (faithfulness) --------------------------------------------
#
# This is ONE LLM call with ONE job: faithfulness. Is every claim in the answer
# grounded in the retrieved CONTEXT? (RAGAS-style claim decomposition; corrects
# by dropping unsupported parts.)
#
# Answerability ("does the surviving answer deliver what the question asked?")
# USED to live here too, but it proved non-deterministic at temperature 0 and
# has moved out to a deterministic regex gate — check_answerability, above —
# wired into the pipeline. Keeping faithfulness alone here means the judge has a
# single, well-posed task it can do reliably.
#
# Why this stays separate from validate_and_fix:
#   - Faithfulness wants the STRONGER mini judge (detecting an ungrounded claim
#     is harder than generating one). validate_and_fix is a profile-safety
#     REWRITE on the nano model with cheap deterministic pre-gates; merging it
#     would force it onto mini, collapse two independent rewrites into one, and
#     lose its zero-LLM-call common case. So: 2 calls, not 1.
#   - This call has no cheap deterministic gate. "Is this claim grounded?" can
#     only be answered by an LLM reading the context, so it's always-on per
#     answer. The only gate is the enabled/disabled config flag.
#
# Why a STRONGER judge model (settings.validator_grounding_model, default
# gpt-4.1-mini) than the nano answer model: detecting an ungrounded claim is
# harder than generating one, and the model that hallucinated is the weakest
# possible grader of its own work.
#
# The corrector is deliberately CONSERVATIVE — it removes only the clause
# carrying an unsupported claim and keeps everything grounded verbatim. Over-
# deletion would make answers terse and drop answer_relevancy (the exact
# faithfulness↔relevancy tension we're managing). The grounded-inference
# allowances below mirror the answer model's own contract so the judge does
# not punish valid inferences (food recommended → safe, "such as" examples,
# etc.) and gut otherwise-correct answers.


class _ClaimVerdict(BaseModel):
    """One atomic claim extracted from the answer, with its grounding verdict."""
    claim: str = Field(
        ...,
        description="A single atomic factual claim taken from the answer.",
    )
    grounded: bool = Field(
        ...,
        description="True if the claim is supported by the context (directly or via an allowed inference).",
    )
    evidence: str = Field(
        ...,
        description="Which part of the context supports the claim, or 'not in context' if unsupported.",
    )


class AnswerReview(BaseModel):
    """
    Output of review_answer (faithfulness only).

    is_grounded: True if EVERY claim in the answer is grounded.
    claims: per-claim decomposition + verdicts (also useful for tracing).
    corrected_answer: the answer text to use downstream.

      IMPORTANT: when is_grounded=True, this field is set to the ORIGINAL
      answer by the calling code, NOT by the LLM — same no-echo-trust safety
      as ValidationResult. The LLM only rewrites when claims are ungrounded.

    (Answerability is no longer judged here — see check_answerability, a
    deterministic regex gate the pipeline applies to corrected_answer.)
    """
    is_grounded: bool
    claims: list[_ClaimVerdict]
    corrected_answer: str


_REVIEW_SYSTEM_PROMPT = """You are a reviewer for a pregnancy-nutrition assistant. You judge whether an ANSWER is fully grounded in the provided CONTEXT. When the answer is not grounded, you produce a corrected answer with the unsupported parts removed.

METHOD (follow it exactly — do not judge by overall impression):
  1. Decompose the answer into atomic claims — one simple factual statement each. Split aggressively; under-splitting is the main way a hallucination escapes review:
     - A sentence with two facts becomes two claims.
     - A clause stating a PURPOSE, BENEFIT, or RESULT — "to support X", "which helps Y", "for Z", "so that…", "to promote…", "to keep you healthy" — becomes its OWN claim, separate from the factual statement it is attached to. A benefit or purpose tacked onto the end of an otherwise-grounded sentence is the single most common place an ungrounded claim hides. Never let a grounded clause carry an ungrounded purpose through as one claim — isolate the purpose and judge it on its own.
     - When a sentence attributes a property to a SET of items ("these foods provide…", "they are rich in…"), the claim is that THOSE SPECIFIC items have that property. Verify the context attaches that property to the SAME items — not merely to some other items somewhere else in the context. Borrowing a property the context stated about a different list is an ungrounded merge.
  2. For each claim, decide whether the CONTEXT supports it.
  3. The answer is_grounded only if EVERY claim is supported.

A claim is GROUNDED if the context supports it directly OR via one of these allowed inferences (these are legitimate, not hallucination):
  - The context recommends a food → that food is safe and beneficial in pregnancy.
  - The context lists items "such as" X, Y → X and Y are examples of that category.
  - The context says something applies "during pregnancy" → it applies to the user.
  - The context gives supplement guidance and the answer reports it as supplement guidance → grounded, even if the question was about food.

A claim is UNGROUNDED (hallucination, even if medically accurate in the real world) if:
  - It states a fact that simply is not in the context.
  - It cites a food for a purpose, nutrient, or benefit the context does not assign to it. The context's framing is the only framing allowed: a food listed under general diet advice cannot be relabeled as "iron-rich" or tied to a specific nutrient the context didn't connect it to.
  - It adds a reason, benefit, or causal link the context does not state. The context saying WHAT to do ("take one tablet daily") does not support a claim about WHY ("to prevent deficiency", "to support foetal development").
  - It merges separate context items (separate bullets, rows, or chunks) into one compound claim neither source makes on its own.
  - It assigns a food, nutrient, or action to a specific time, meal, or occasion (breakfast/lunch/dinner, morning/night) that the context does not explicitly tie it to. The context listing foods for "the daily diet" PLUS a separate statement that meals divide into breakfast/lunch/dinner does NOT license placing specific foods in specific meals — that pairing is compound synthesis.

A DEFLECTION IS NOT GROUNDED CONTENT. A meta-statement about the guidelines themselves — "the guidelines do not specify X", "the context does not mention Y", "there is no information about Z" — carries no nutrition content and must NOT be treated as a grounded claim worth keeping. It is the ABSENCE of an answer, not an answer.

CORRECTION (only when is_grounded is false):
  - Remove ONLY the clause or sentence carrying each ungrounded claim. Keep every grounded claim verbatim — do not paraphrase, reorder, or shorten grounded content.
  - Add nothing. Do not introduce new facts, and do not add a remark about what the context "does not mention" — just drop the unsupported part cleanly.
  - If removing leaves a partial sentence, smooth it minimally (e.g. drop a trailing item from a list) without changing meaning.
  - Preserve the original voice, tone, and length of what remains.
  - If, after removing every ungrounded claim, the ONLY thing left would be a deflection (see above) or nothing at all, then the answer had no grounded content to begin with. In that case set corrected_answer to the EMPTY STRING "" — do NOT keep the deflection, and do NOT invent a replacement. The calling code converts an empty result into an honest "I don't have that information" response.

OUTPUT:
  - is_grounded: true only if every claim is grounded.
  - claims: the decomposition with per-claim grounded/evidence.
  - corrected_answer: when is_grounded is true, set this to the EMPTY STRING "" — the calling code reuses the original answer, so do NOT echo it. When is_grounded is false, set it to the corrected text, OR the empty string "" if nothing grounded remains (see CORRECTION).
"""


def _is_pure_deflection(text: str) -> bool:
    """
    True if EVERY sentence in `text` is a deflective meta-statement (matches a
    forbidden-opener pattern). Used after grounding correction: if the only
    thing the grounding pass left behind is deflection ("the guidelines do not
    specify..."), the answer had no grounded content and must become an honest
    no_results — NOT a kept hedge (which the downstream validate_and_fix would
    then "rewrite" by fabricating substance to replace, the exact failure this
    guards against).
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return False
    return all(
        any(pat.search(s[:120]) for pat in _FORBIDDEN_OPENER_PATTERNS)
        for s in sentences
    )


# --- Deterministic answerability gate ----------------------------------------
#
# Replaces the old LLM answerability verdict (removed from review_answer in
# v3.0). That soft-prompt judgment oscillated TRUE/FALSE across runs at
# temperature 0, and layering more prose rules did not fix it — the same
# "soft rules bypass → enforce deterministically" pattern the diet and opener
# checks already follow.
#
# Scope is deliberately narrow. Across the whole eval suite, the ONLY answers
# that must route to no_results on answerability grounds are QUANTITY questions
# answered with NO quantity (e.g. "how much water should I drink?" answered by
# listing beverages, with no amount). So the rule collapses to exactly that:
#
#   A quantity question ("how much / how many / what amount/quantity") whose
#   answer conveys NO amount of any kind → unanswerable.
#
# Everything else is answerable:
#   - A non-quantity question (a meal-plan ask, a "which foods" ask) never trips
#     the gate — it returns True regardless of the answer.
#   - A quantity question answered with a number, a spelled count, OR a
#     directional amount ("limit it", "reduce intake" — the salt/hypertension
#     case) is answerable.
#
# False negatives (a genuine non-answer we let through) are acceptable — they
# cost a slightly-off answer, not a safety failure. The gate is intentionally
# conservative so it never suppresses a legitimate answer.

_QUANTITY_QUESTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bhow much\b",
        r"\bhow many\b",
        r"\bwhat(?:'?s| is)? the (?:amount|quantity)\b",
        r"\bwhat (?:amount|quantity)\b",
    )
]

# Signals that an answer actually conveys an AMOUNT. Any single match is enough.
#
# Deliberately does NOT match bare measure/serving units ("glasses", "cups").
# A unit word only means "amount" when attached to a number — and the digit
# pattern already catches "8 glasses" / "2 litres". On its own a unit word is
# noise: it shows up echoed from the question ("how many GLASSES") and, worse,
# inside deflections ("the number of glasses is NOT PROVIDED"), which is exactly
# how the water_intake non-answer slipped through as "answerable". So magnitude
# must come from a digit, a spelled-out count, or a directional term.
_AMOUNT_PRESENCE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\d",  # any digit: "2-3 litres", "1000 mg", "8 glasses"
        # spelled-out counts
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|half|dozen)\b",
        # directional / qualitative amounts — answer the "how much" axis without a number
        r"\b(?:limit|reduce|restrict|increase|avoid|minimi[sz]e|fewer|less|more|"
        r"moderate|adequate|enough|plenty|sufficient|extra)\b",
    )
]


def _is_quantity_question(question: str) -> bool:
    return any(pat.search(question) for pat in _QUANTITY_QUESTION_PATTERNS)


def _answer_states_amount(answer: str) -> bool:
    return any(pat.search(answer) for pat in _AMOUNT_PRESENCE_PATTERNS)


def check_answerability(question: str, answer: str) -> bool:
    """
    Deterministic answerability gate. Returns True when the answer is considered
    to answer the question, False when it should route to no_results.

    The ONLY False case is a quantity question ("how much / how many / what
    amount") whose answer conveys no amount at all — no number, no unit, no
    spelled count, and no directional amount. Every other question/answer pair
    returns True; the gate does not second-guess non-quantity questions.
    """
    if not answer.strip():
        return False
    if not _is_quantity_question(question):
        return True
    return _answer_states_amount(answer)


@observe(name="review_answer")
def review_answer(answer: str, context: str, question: str) -> AnswerReview:
    """
    Review `answer` for faithfulness in a single LLM call: every claim must be
    grounded in `context`; ungrounded parts are dropped from corrected_answer.

    (Answerability is no longer judged here — the pipeline applies the
    deterministic check_answerability gate to corrected_answer instead.)

    `context` is the SAME formatted context string the answer LLM saw — the
    judge evaluates against exactly what the model was shown. `question` is the
    original user message (still accepted for the prompt's framing).

    Gate: when settings.validator_grounding_enabled is False, returns the
    original answer with zero LLM calls (used to A/B the feature in eval).

    No-op safety: when is_grounded comes back True, the CODE returns the
    original answer — we never trust the LLM to echo ~1KB verbatim. Same
    rationale as validate_and_fix (see MEMORY.md "Don't trust LLMs to ECHO").

    Empty result contract: when the answer was ungrounded AND nothing grounded
    survives correction (LLM returned empty, or left only a deflection),
    corrected_answer is the EMPTY STRING. The pipeline treats that as the
    signal to emit NO_RESULTS_RESPONSE — an honest "I don't have that" beats
    re-surfacing the hallucination or letting validate_and_fix fabricate a
    replacement for a deflection-only answer.
    """
    if not settings.validator_grounding_enabled:
        update_current_span(
            metadata={"review_skipped": "disabled"},
            output={"is_grounded": True, "corrected": False},
        )
        return AnswerReview(
            is_grounded=True, claims=[], corrected_answer=answer,
        )

    update_current_span(input={
        "original_answer": answer,
        "question": question,
        "judge_model": settings.validator_grounding_model,
        "prompt_version": REVIEW_PROMPT_VERSION,
    })

    response = get_openai_client().beta.chat.completions.parse(
        model=settings.validator_grounding_model,
        temperature=0,
        messages=[
            {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": (
                "QUESTION:\n" + question
                + "\n\nCONTEXT:\n" + context
                + "\n\nANSWER:\n" + answer
            )},
        ],
        response_format=AnswerReview,
    )
    raw = response.choices[0].message.parsed
    if raw is None:
        raise RuntimeError(
            "Answer reviewer returned None — structured output parse failed."
        )

    # No-op safety: CODE owns the echo when grounded. When ungrounded, use the
    # LLM's rewrite — but if nothing grounded survives (empty rewrite, or a
    # rewrite that is nothing but deflection), emit the EMPTY STRING so the
    # pipeline can route to no_results. We deliberately do NOT fall back to the
    # original answer (that re-surfaces the hallucination) and do NOT keep a
    # deflection (validate_and_fix would then fabricate substance to replace
    # it — the water_intake failure this fixes).
    if raw.is_grounded:
        final_answer = answer
    else:
        rewritten = raw.corrected_answer.strip()
        if not rewritten or _is_pure_deflection(rewritten):
            final_answer = ""
        else:
            final_answer = rewritten

    result = AnswerReview(
        is_grounded=raw.is_grounded,
        claims=list(raw.claims),
        corrected_answer=final_answer,
    )

    n_ungrounded = sum(1 for c in result.claims if not c.grounded)
    update_current_span(
        output={
            "is_grounded": result.is_grounded,
            "corrected_answer": result.corrected_answer,
        },
        metadata={
            "n_claims": len(result.claims),
            "n_ungrounded": n_ungrounded,
            "answer_changed": result.corrected_answer != answer,
            "claims": [c.model_dump() for c in result.claims],
        },
    )
    return result
