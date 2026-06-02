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

PROMPT_VERSION = "v2.1"  # v2.1: added trailing-deflection detection + rule

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

    if not dietary_rules and not has_opener_issue and not has_trailing_issue:
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
    rules.extend(dietary_rules)

    update_current_span(input={
        "n_rules": len(rules),
        "has_opener_issue_pre_check": has_opener_issue,
        "has_trailing_issue_pre_check": has_trailing_issue,
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
