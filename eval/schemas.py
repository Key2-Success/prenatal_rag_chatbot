"""
schemas.py — Pydantic schemas for the v1 evaluation suite.

Why schemas (not raw dicts):
  YAML is convenient to edit but offers no type safety. Parsing into Pydantic
  models gives us, at load time:
    - typo detection with line-level error messages
    - enum-constrained fields (no string-typing for behavior, category, etc.)
    - cross-field validation (cites_org only valid when behavior=answer)
    - cross-document validation (every case.profile exists, no duplicate ids)
    - one source of truth: valid orgs come from sources.json, not a hardcode

Reuse, don't duplicate:
  - UserProfile is reused from backend.app.models.schemas — same model the
    API uses, so app/eval drift is impossible by construction.
  - ResponseType is reused as the expected behaviour enum. Eval tests "did
    the pipeline produce response_type=X" — the enum and
    ChatResponse.response_type are the same shape, so we share one definition.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.schemas import ResponseType, UserProfile
from backend.app.sources import priority_order


class StrictUserProfile(UserProfile):
    """
    Eval-only subclass of UserProfile that rejects unknown fields.

    The API's UserProfile uses Pydantic's default `extra="ignore"` so that
    clients sending extra fields don't get 422'd. In eval YAML that
    permissiveness is a footgun — a typo like `medical_condition: [Low iron]`
    (missing 's') would be silently dropped and the eval would run with an
    empty conditions list. `extra="forbid"` turns those typos into a load-
    time ValidationError naming the offending field.
    """
    model_config = ConfigDict(extra="forbid")


class Category(str, Enum):
    """Test-case category — used for grouped reporting only."""
    core_nutrition = "core_nutrition"
    personalization = "personalization"
    indian_context = "indian_context"
    guardrail = "guardrail"
    out_of_scope = "out_of_scope"
    edge_case = "edge_case"


class ExpectedOutcome(BaseModel):
    """The `expected:` block on each test case."""
    behavior: ResponseType
    cites_org: str | None = None

    @model_validator(mode="after")
    def _validate_cites_org(self) -> "ExpectedOutcome":
        # Asserting a citation only makes sense when an answer was produced.
        # Emergencies and out-of-scope responses don't carry sources.
        if self.cites_org is None:
            return self
        if self.behavior is not ResponseType.answer:
            raise ValueError(
                f"cites_org is only valid when behavior=answer "
                f"(got behavior={self.behavior.value})"
            )
        # Pull valid orgs from sources.json — single source of truth. Adding
        # a new PDF in data/sources.json automatically expands the allowed set.
        valid_orgs = priority_order()
        if self.cites_org not in valid_orgs:
            raise ValueError(
                f"cites_org must be one of {list(valid_orgs)}, got {self.cites_org!r}"
            )
        return self


class TestCase(BaseModel):
    """One row of test_cases.yaml."""
    id: str = Field(..., min_length=1, max_length=80)
    category: Category
    query: str = Field(..., min_length=1, max_length=1000)
    profile: str = Field(..., min_length=1)  # key into the profile registry
    expected: ExpectedOutcome


class EvalSuite(BaseModel):
    """
    The fully validated evaluation suite — both files parsed together so we
    can enforce cross-document invariants (profile references, unique ids).
    """
    profiles: dict[str, StrictUserProfile]
    cases: list[TestCase]

    @model_validator(mode="after")
    def _validate_references(self) -> "EvalSuite":
        # Unique case ids — duplicates would silently overwrite each other in
        # the markdown report.
        seen: set[str] = set()
        dups: set[str] = set()
        for c in self.cases:
            if c.id in seen:
                dups.add(c.id)
            seen.add(c.id)
        if dups:
            raise ValueError(f"duplicate test case ids: {sorted(dups)}")

        # Every case must reference a real profile, or we'd KeyError at run time.
        unknown = {c.profile for c in self.cases} - set(self.profiles)
        if unknown:
            raise ValueError(
                f"test cases reference unknown profiles: {sorted(unknown)}. "
                f"Known: {sorted(self.profiles)}"
            )
        return self
