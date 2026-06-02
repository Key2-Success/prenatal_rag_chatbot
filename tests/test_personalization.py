"""
test_personalization.py — Schema-driven personalization invariants.

Covers the parts of the personalization architecture that aren't enforced by
__init_subclass__ at import time:

  - TrimesterRule: range-based logic produces a non-empty rule for every
    week in the legal range [1, 45]. The else branch in to_prompt_rule
    handles the open end, but a forgotten branch could still produce empty
    or missing text.
  - UserProfile.to_personalization_block: assembles rules in the documented
    order (diet → conditions → trimester), one bullet per provider.
  - PreferenceEnum runtime invariant: every member of every preference enum
    returns a non-empty string from to_prompt_rule. This is ALSO checked at
    class-definition time by __init_subclass__, but a parametrized test
    documents the contract explicitly and catches regressions if the
    enforcement is ever loosened.

What's NOT tested here:
  - The class-definition-time enforcement itself (we can't easily test
    "module fails to import" inside the same test module without
    sub-process gymnastics). The behaviour is exercised every time anyone
    imports schemas.py — the app refuses to start if a rule is missing.
"""

import pytest

from backend.app.models.schemas import (
    DietType,
    MedicalCondition,
    PreferenceEnum,
    TrimesterRule,
    UserProfile,
)


# ---------- TrimesterRule ----------

@pytest.mark.parametrize(
    "week,expected_trimester_substring",
    [
        # Boundaries of each trimester. The branching logic uses week <= 12,
        # week <= 26, else — so 12/13 and 26/27 are the critical transitions.
        (1, "1st trimester"),
        (12, "1st trimester"),
        (13, "2nd trimester"),
        (26, "2nd trimester"),
        (27, "3rd trimester"),
        (45, "3rd trimester"),  # Pydantic's upper bound on pregnancy_week
    ],
)
def test_trimester_rule_returns_non_empty_with_correct_label(
    week: int, expected_trimester_substring: str
) -> None:
    rule = TrimesterRule(week=week).to_prompt_rule()
    assert rule.strip(), f"Rule for week {week} is empty"
    assert expected_trimester_substring in rule, (
        f"Rule for week {week} doesn't identify the right trimester: {rule!r}"
    )
    assert f"week {week}" in rule, (
        f"Rule for week {week} doesn't include the week number: {rule!r}"
    )


# ---------- Preference enum coverage ----------

@pytest.mark.parametrize(
    "enum_cls",
    [DietType, MedicalCondition],
    ids=lambda c: c.__name__,
)
def test_every_preference_enum_value_has_non_empty_rule(
    enum_cls: type[PreferenceEnum],
) -> None:
    """
    Every member of every PreferenceEnum subclass returns a non-empty rule.

    Belt-and-suspenders with the __init_subclass__ check — that runs at
    import time; this is the documented contract test that survives future
    refactors of the enforcement mechanism.
    """
    for member in enum_cls:
        rule = member.to_prompt_rule()
        assert isinstance(rule, str), (
            f"{enum_cls.__name__}.{member.name}.to_prompt_rule() returned "
            f"non-string: {type(rule).__name__}"
        )
        assert rule.strip(), (
            f"{enum_cls.__name__}.{member.name}.to_prompt_rule() is empty"
        )


# ---------- UserProfile assembly ----------

def _make_profile(
    diet: DietType,
    week: int,
    conditions: list[MedicalCondition],
) -> UserProfile:
    """Tiny builder so each test case is one readable line."""
    return UserProfile(
        name="Test",
        age=28,
        pregnancy_week=week,
        diet_type=diet,
        weight_kg=60.0,
        height_cm=160.0,
        medical_conditions=conditions,
    )


def test_personalization_block_is_bulleted_one_line_per_rule() -> None:
    profile = _make_profile(
        diet=DietType.vegetarian,
        week=20,
        conditions=[MedicalCondition.low_iron],
    )
    block = profile.to_personalization_block()
    lines = block.splitlines()
    # 1 diet + 1 condition + 1 trimester = 3 bullets
    assert len(lines) == 3, f"expected 3 bullets, got {len(lines)}: {block!r}"
    for line in lines:
        assert line.startswith("- "), f"bullet missing dash prefix: {line!r}"


def test_personalization_block_orders_diet_then_conditions_then_trimester() -> None:
    profile = _make_profile(
        diet=DietType.ovo_vegetarian,
        week=30,  # 3rd trimester
        conditions=[MedicalCondition.diabetes, MedicalCondition.hypertension],
    )
    lines = profile.to_personalization_block().splitlines()
    # Diet first, then conditions in profile order, then trimester last.
    assert "Ovo-Vegetarian" in lines[0]
    assert "diabetes" in lines[1].lower()
    assert "hypertension" in lines[2].lower()
    assert "3rd trimester" in lines[3] if len(lines) > 3 else False
    # That last assert is structured oddly to make the failure mode clear if
    # we ever add a new rule provider — len(lines) WILL change and the test
    # should catch it.
    assert len(lines) == 4


def test_personalization_block_handles_no_medical_conditions() -> None:
    profile = _make_profile(
        diet=DietType.non_vegetarian,
        week=8,  # 1st trimester
        conditions=[],
    )
    lines = profile.to_personalization_block().splitlines()
    # 1 diet + 0 conditions + 1 trimester = 2 bullets
    assert len(lines) == 2
    assert "Non-Vegetarian" in lines[0]
    assert "1st trimester" in lines[1]


# ---------- Validation rule contracts ----------

@pytest.mark.parametrize(
    "diet,expected_rule_present",
    [
        (DietType.vegetarian, True),
        (DietType.ovo_vegetarian, True),
        (DietType.non_vegetarian, False),  # No restrictions for non-veg
    ],
    ids=lambda x: getattr(x, "name", str(x)),
)
def test_diet_validation_rule_returns_string_or_none(
    diet: DietType, expected_rule_present: bool
) -> None:
    rule = diet.to_validation_rule()
    if expected_rule_present:
        assert isinstance(rule, str) and rule.strip(), (
            f"{diet.name} should have a validation rule"
        )
    else:
        assert rule is None, (
            f"{diet.name} should have no validation rule (got {rule!r})"
        )


@pytest.mark.parametrize(
    "condition,expected_rule_present",
    [
        (MedicalCondition.low_iron, False),     # Emphasis, not restriction
        (MedicalCondition.hypertension, True),  # Sodium restriction
        (MedicalCondition.diabetes, True),       # Sugar/refined-carb restriction
    ],
    ids=lambda x: getattr(x, "name", str(x)),
)
def test_condition_validation_rule_returns_string_or_none(
    condition: MedicalCondition, expected_rule_present: bool
) -> None:
    rule = condition.to_validation_rule()
    if expected_rule_present:
        assert isinstance(rule, str) and rule.strip(), (
            f"{condition.name} should have a validation rule"
        )
    else:
        assert rule is None, (
            f"{condition.name} should have no validation rule (got {rule!r})"
        )


# ---------- Validator integration (mocked LLM) ----------

def test_validator_short_circuits_when_no_rules_apply(monkeypatch) -> None:
    """
    Non-veg + no condition-driven restrictions → zero LLM calls. The validator
    returns the original answer unchanged. This is the cost-optimisation path.
    """
    from backend.app.chat import validator as v

    # Sentinel that would fail loudly if the LLM is called by accident.
    def _should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "OpenAI client called for a profile with no validation rules"
        )

    monkeypatch.setattr(
        "backend.app.chat.validator.get_openai_client",
        _should_not_be_called,
    )

    profile = _make_profile(
        diet=DietType.non_vegetarian,
        week=20,
        conditions=[MedicalCondition.low_iron],  # Low iron is emphasis, not restriction
    )
    original = "Good protein sources for you include chicken, fish, eggs, and dal."
    result = v.validate_and_fix(original, profile)

    assert result.is_compliant is True
    assert result.violations == []
    assert result.corrected_answer == original


def _mock_openai_for_validator(monkeypatch, parsed: object) -> None:
    """
    Helper that mocks get_openai_client() in the validator module so the
    structured-output call returns a fixed `parsed` value. Lets each test
    set the validator's classification + correction outputs.
    """
    from backend.app.chat import validator as v

    fake_choice = type("FakeChoice", (), {
        "message": type("FakeMsg", (), {"parsed": parsed})(),
    })()
    fake_response = type("FakeResponse", (), {"choices": [fake_choice]})()

    class _Completions:
        def parse(self, **_kwargs):
            return fake_response

    class _Chat:
        completions = _Completions()

    class _Beta:
        chat = _Chat()

    class _Client:
        beta = _Beta()

    monkeypatch.setattr(
        "backend.app.chat.validator.get_openai_client",
        lambda: _Client(),
    )


def test_validator_returns_corrected_answer_when_violations_found(monkeypatch) -> None:
    """Violation path: LLM-provided corrected_answer flows through to caller."""
    from backend.app.chat import validator as v

    expected_corrected = "Good protein sources for you include eggs, paneer, and dal."
    _mock_openai_for_validator(monkeypatch, parsed=v.ValidationResult(
        is_compliant=False,
        violations=[
            v._Violation(
                field="diet_type",
                violating_foods=["chicken", "fish"],
                explanation="Ovo-vegetarian — chicken and fish not allowed.",
            ),
        ],
        corrected_answer=expected_corrected,
    ))

    profile = _make_profile(
        diet=DietType.ovo_vegetarian, week=20, conditions=[],
    )
    original = "Good protein sources for you include eggs, paneer, dal, chicken, and fish."
    result = v.validate_and_fix(original, profile)

    assert result.is_compliant is False
    assert len(result.violations) == 1
    assert result.corrected_answer == expected_corrected
    assert "chicken" not in result.corrected_answer.lower()
    assert "fish" not in result.corrected_answer.lower()


def test_validator_uses_original_when_compliant_even_if_llm_returns_empty(monkeypatch) -> None:
    """
    No-op safety path: when is_compliant=True, CODE returns the original
    answer regardless of what the LLM put in corrected_answer. This guards
    against the historical bug where the LLM returned `corrected_answer="{"`
    for a perfectly compliant long answer (see MEMORY.md).
    """
    from backend.app.chat import validator as v

    original = (
        "For breakfast, include whole grains like oats with vitamin C rich "
        "foods such as lemon or tomato, and add nuts or seeds for extra iron. "
        "For lunch and dinner, consume green leafy vegetables, whole pulses, "
        "and whole grains, along with fruits like guava or orange, and include "
        "milk or milk products."
    )

    # Simulate the LLM returning is_compliant=True but a MALFORMED
    # corrected_answer ("{" — the exact bug that triggered this fix).
    _mock_openai_for_validator(monkeypatch, parsed=v.ValidationResult(
        is_compliant=True,
        violations=[],
        corrected_answer="{",  # broken — code must NOT trust this
    ))

    profile = _make_profile(
        diet=DietType.vegetarian,  # has rules → validator runs
        week=28,
        conditions=[MedicalCondition.low_iron],
    )
    result = v.validate_and_fix(original, profile)

    assert result.is_compliant is True
    assert result.violations == []
    # The critical assertion: corrected_answer is the ORIGINAL, set by code,
    # NOT the LLM's broken "{". Without the code-enforced no-op, this would
    # be "{" and the user would see one character.
    assert result.corrected_answer == original


def test_validator_falls_back_to_original_when_llm_returns_empty_on_violation(monkeypatch) -> None:
    """
    Defensive path: even on the violation branch, if the LLM returns an
    empty corrected_answer (it shouldn't but might), we keep the original
    rather than emit blank output to the user.
    """
    from backend.app.chat import validator as v

    original = "Some protein sources include eggs, dal, chicken."

    _mock_openai_for_validator(monkeypatch, parsed=v.ValidationResult(
        is_compliant=False,
        violations=[
            v._Violation(field="diet_type", violating_foods=["chicken"],
                         explanation="Vegetarian — chicken not allowed."),
        ],
        corrected_answer="   ",  # whitespace = effectively empty
    ))

    profile = _make_profile(
        diet=DietType.vegetarian, week=20, conditions=[],
    )
    result = v.validate_and_fix(original, profile)

    # Validator's defensive fallback kicks in — original is preserved
    # rather than blank output reaching the user.
    assert result.corrected_answer == original


# ---------- Deflective opener detection ----------

@pytest.mark.parametrize("answer,expected", [
    # Forbidden openers — all should be detected
    ("The guidelines do not specify an exact amount of paneer.", True),
    ("The guidelines don't specify a specific amount.", True),
    ("The context does not mention amla.", True),
    ("The context doesn't say.", True),
    ("There is no specific guidance.", True),
    ("There are no specific recommendations.", True),
    ("I don't have that information.", True),
    ("I do not have specific data.", True),
    ("Unfortunately, the guidelines don't list a specific amount.", True),
    ("While the guidelines don't specify exact amounts, you can include...", True),
    ("Although the guidelines don't state precise quantities...", True),
    ("The provided guidelines do not list folate-rich foods.", True),
    # Whitespace before forbidden phrase — still detected
    ("   The guidelines do not specify...", True),
    # Clean openers — should NOT be detected (these are good)
    ("Yes, amla is safe to eat during pregnancy.", False),
    ("Good protein sources for you include dal and paneer.", False),
    ("The guidelines recommend 100 mg of elemental iron daily.", False),
    ("During pregnancy, you should consume 1g of calcium daily.", False),
    # Edge: phrase appears NOT at the start
    ("Iron is important. The guidelines do not specify exact amounts.", False),
    # Edge: empty / whitespace input
    ("", False),
    ("   ", False),
])
def test_deflective_opener_detection(answer: str, expected: bool) -> None:
    from backend.app.chat.validator import _has_deflective_opener
    assert _has_deflective_opener(answer) == expected, (
        f"detection wrong for: {answer!r}"
    )


def test_validator_short_circuits_when_no_rules_AND_clean_opener(monkeypatch) -> None:
    """
    The cost-optimisation short-circuit fires when BOTH conditions are met:
    no profile rules apply AND the answer has no deflective opener.
    """
    from backend.app.chat import validator as v

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("OpenAI called when both gates were clear")

    monkeypatch.setattr(
        "backend.app.chat.validator.get_openai_client", _should_not_be_called,
    )

    profile = _make_profile(
        diet=DietType.non_vegetarian, week=20,
        conditions=[MedicalCondition.low_iron],  # emphasis, not restriction
    )
    clean_answer = "Iron supplementation of 100 mg daily is recommended from the second trimester."
    result = v.validate_and_fix(clean_answer, profile)

    assert result.is_compliant is True
    assert result.violations == []
    assert result.corrected_answer == clean_answer


def test_validator_invokes_llm_for_opener_even_when_no_profile_rules(monkeypatch) -> None:
    """
    Even with zero profile rules (non-veg + low iron), a detected deflective
    opener triggers the validator's LLM call so the opener can be rewritten.
    """
    from backend.app.chat import validator as v

    rewritten = "Include protein-rich foods like paneer in your meals. The guidelines don't list an exact daily amount per food."
    _mock_openai_for_validator(monkeypatch, parsed=v.ValidationResult(
        is_compliant=False,
        violations=[
            v._Violation(
                field="opener",
                violating_foods=[],
                explanation="Answer opens with 'The guidelines do not specify...'",
            ),
        ],
        corrected_answer=rewritten,
    ))

    profile = _make_profile(
        diet=DietType.non_vegetarian, week=20, conditions=[],
    )
    bad_opener = "The guidelines do not specify an exact amount of paneer. However, you can include protein-rich foods in your meals."
    result = v.validate_and_fix(bad_opener, profile)

    assert result.is_compliant is False
    assert result.violations[0].field == "opener"
    assert result.corrected_answer == rewritten
    # The corrected answer must NOT start with the forbidden phrase
    assert not result.corrected_answer.lower().startswith("the guidelines do not")


# ---------- Trailing deflection detection ----------

@pytest.mark.parametrize("answer,expected", [
    # Trailing deflection — should be detected
    ("Drink 2-3 liters of water per day. The guidelines do not specify an exact number of glasses.", True),
    ("Include one serving of protein-rich foods like paneer. The guidelines don't list an exact daily amount per food.", True),
    ("Iron supplementation of 100 mg daily is recommended. There is no specific upper limit mentioned.", True),
    ("Take folic acid daily. I don't have information on specific brands.", True),
    # Clean trailing — should NOT be detected
    ("Drink 2-3 liters of water per day. This helps keep the bowels regular.", False),
    ("Include one serving of protein-rich foods like paneer per meal.", False),
    ("Iron supplementation of 100 mg daily is recommended from the second trimester.", False),
    # Single-sentence forbidden phrase — opener check handles, trailing returns False
    # (so we don't double-flag the same sentence)
    ("The guidelines do not specify an exact amount.", False),
    # Edge: empty / whitespace
    ("", False),
    ("   ", False),
    # Edge: phrase in MIDDLE, not at end
    ("The guidelines do not specify exactly. However, eat 60mg of iron daily.", False),
])
def test_trailing_deflection_detection(answer: str, expected: bool) -> None:
    from backend.app.chat.validator import _has_trailing_deflection
    assert _has_trailing_deflection(answer) == expected, (
        f"detection wrong for: {answer!r}"
    )


def test_validator_invokes_llm_for_trailing_deflection(monkeypatch) -> None:
    """
    A clean opener + a trailing deflection still triggers the validator
    (because trailing was detected). LLM rewrites by removing the last
    sentence; the rest is unchanged.
    """
    from backend.app.chat import validator as v

    rewritten = "Drink approximately 2-3 liters of water per day to keep the bowels regular."
    _mock_openai_for_validator(monkeypatch, parsed=v.ValidationResult(
        is_compliant=False,
        violations=[
            v._Violation(
                field="trailing",
                violating_foods=[],
                explanation="Last sentence was a useless hedge.",
            ),
        ],
        corrected_answer=rewritten,
    ))

    profile = _make_profile(
        diet=DietType.non_vegetarian, week=20, conditions=[],
    )
    bad_trailing = (
        "Drink approximately 2-3 liters of water per day to keep the bowels regular. "
        "The guidelines do not specify an exact number of glasses."
    )
    result = v.validate_and_fix(bad_trailing, profile)

    assert result.is_compliant is False
    assert result.violations[0].field == "trailing"
    assert result.corrected_answer == rewritten
    # Trailing sentence must be gone from the final answer.
    assert "do not specify" not in result.corrected_answer.lower()


def test_validator_regex_overrides_llm_when_llm_misses_trailing(monkeypatch) -> None:
    """
    If the LLM incorrectly returns is_compliant=True when regex detected a
    trailing deflection, the code overrides — adds the trailing violation
    and uses the corrected_answer (or original if blank).
    """
    from backend.app.chat import validator as v

    # LLM returns "compliant" despite the regex pre-check flagging trailing.
    # The LLM ALSO returns a sane corrected_answer (defensive — testing the
    # override path independently of the rewrite path).
    rewritten_by_llm = "Drink 2-3 liters of water per day."
    _mock_openai_for_validator(monkeypatch, parsed=v.ValidationResult(
        is_compliant=True,  # ← LLM disagreed with regex
        violations=[],
        corrected_answer=rewritten_by_llm,
    ))

    profile = _make_profile(
        diet=DietType.non_vegetarian, week=20, conditions=[],
    )
    bad_trailing = (
        "Drink 2-3 liters of water per day. "
        "The guidelines do not specify an exact number of glasses."
    )
    result = v.validate_and_fix(bad_trailing, profile)

    # Override: code says NOT compliant because regex detected the trailing.
    assert result.is_compliant is False
    assert any(v.field == "trailing" for v in result.violations)


def test_validator_short_circuits_when_all_three_gates_clean(monkeypatch) -> None:
    """
    No profile rules, no opener issue, no trailing issue → zero LLM calls.
    """
    from backend.app.chat import validator as v

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("OpenAI called when all three gates were clear")

    monkeypatch.setattr(
        "backend.app.chat.validator.get_openai_client", _should_not_be_called,
    )

    profile = _make_profile(
        diet=DietType.non_vegetarian, week=20, conditions=[],
    )
    clean = (
        "Iron supplementation of 100 mg daily is recommended from the second "
        "trimester onwards for at least 180 days."
    )
    result = v.validate_and_fix(clean, profile)

    assert result.is_compliant is True
    assert result.corrected_answer == clean
