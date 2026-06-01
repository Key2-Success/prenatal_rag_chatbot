"""
hyde.py — HyDE (Hypothetical Document Embeddings) query transformation.

The idea (from Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without
Relevance Labels"): natural-language QUESTIONS embed differently than the
prose ANSWERS that should retrieve for them. A user asks "how much iron
should I take?" — the answer chunk says "Pregnant women should consume
60 mg of elemental iron daily..." — these don't match in embedding space
nearly as well as you'd hope, because the embedding model is trained on
prose-style text, not Q&A pairs.

HyDE fixes this by generating a HYPOTHETICAL answer to the question with
a small LLM, then embedding THAT instead of the question. The hypothetical
answer is prose, often factually similar to the real answer, and matches
real answer chunks far better.

Trade-offs:
  + Substantially improves retrieval recall on natural-language queries,
    especially with prose chunks (validated in the HyDE paper across MS
    MARCO, BEIR, mixed-domain benchmarks).
  + Closes the prose↔question semantic gap.
  - Adds one LLM call per query (~50-200ms latency, ~$0.0001 cost).
  - Wrong-but-confident hypothetical answers can retrieve wrong chunks
    (mitigated by the "for embedding only" framing in the prompt).
  - Won't help table-shaped chunks much — those need a separate fix
    (table summarisation at ingest, planned as Phase 2).

This module ONLY generates hypothetical answers; embedding and retrieval
stay in their existing modules (embedder.py, retriever.py).

Profile personalisation:
  The hypothetical answer is generated WITH the user's profile context
  (pregnancy week, diet, conditions). This means the hypothetical answer
  is tailored to the user — a vegetarian asking about protein gets a
  plant-based hypothetical answer that matches plant-based chunks. Without
  the profile, HyDE would generate generic answers that hurt personalised
  retrieval more than it helps.

Prompt versioning:
  PROMPT_VERSION is a code-versioned (git-tracked) identifier. Bump on
  every meaningful prompt change so reports and traces can be tied back to
  the exact prompt that produced them. Future work: migrate to Langfuse
  Prompt Management for runtime A/B testing without redeploys (see plan).
"""

from backend.app.clients import get_openai_client
from backend.app.config import settings
from backend.app.models.schemas import UserProfile
from backend.app.observability import observe, update_current_span

# Bump on every meaningful prompt change. Use semver-ish: v1.0 → v1.1 for
# wording tweaks, v1.x → v2.0 for structural changes (added/removed fields,
# different output format). Older versions live in git history.
#
# v1.0 → v2.0 (2026-05-23): structural rewrite after v1.0 tanked the eval.
# Two failures fixed:
#   1. Wrong voice target. v1.0 said "simulate a Poshan Saathi answer" —
#      that embeds near chatty user-facing answer prose that doesn't exist
#      in the index. v2.0 targets clinical-guideline-document voice (third
#      person, formal, dense with clinical facts) — that's what's actually
#      IN the corpus (MoHFW/FOGSI/WHO).
#   2. Profile constraints not enforced. v1.0 said "tailor to diet" — too
#      vague, LLM produced non-veg sources for vegetarian users. v2.0 ports
#      the explicit DIET FILTER + worked example pattern from pipeline.py's
#      SYSTEM_PROMPT, which we already learned was necessary for the
#      production answer LLM. Same rigor, different voice.
PROMPT_VERSION = "v2.0"

# The "for embedding only, never shown" framing is load-bearing — it lets
# the LLM produce confident specific text without the safety hedging that
# would dominate the embedding signal. Hedged prose like "you should
# consult your doctor about iron" embeds far from the actual answer chunk
# "Pregnant women should consume 60 mg of elemental iron daily."
_SYSTEM_PROMPT = """You are generating a passage from a clinical antenatal-care guideline document that would answer the user's question, using ONLY foods and recommendations that fit the user's diet and medical context.

This passage will ONLY be used as a search-query embedding to retrieve relevant guideline chunks — it will NEVER be shown to a user. So:

- Be specific: state concrete dosages, durations, timing, food groups, clinical thresholds (e.g., haemoglobin levels for anaemia diagnosis)
- Use formal, third-person, authoritative language ("Pregnant women should...", "Iron supplementation is recommended..."). Do NOT use second-person or conversational language.
- 1-2 paragraphs, dense with clinical facts
- No safety caveats, no hedges, no markdown, no bullet lists

DIET FILTER — apply before mentioning any food:
- Vegetarian → ONLY plant-based foods. Omit meat, poultry, fish, eggs.
- Ovo-Vegetarian → plant-based foods + eggs only. Omit meat, poultry, fish.
- Non-Vegetarian → all foods are acceptable.

Example — User diet = Vegetarian, question = "What are some good protein sources?":
Correct: "Pregnant vegetarian women meet protein requirements through legumes, dal, paneer, tofu, and dairy products. The recommended daily intake is 71 g during the second and third trimesters."
Wrong:   "Protein sources include chicken, fish, eggs, and dal." (Includes non-veg items for a vegetarian user — filter violation.)

MEDICAL CONDITION HANDLING:
- Low iron → emphasize iron-rich foods compatible with the user's diet, iron supplementation thresholds (100 mg elemental iron daily), and haemoglobin diagnostic cut-offs.
- Hypertension → emphasize sodium restriction, DASH-style guidance.
- Diabetes (gestational) → emphasize carbohydrate quality, glycemic load.

PREGNANCY WEEK / TRIMESTER:
- Weeks 1-12 (1st trimester) → folic acid emphasis (500 mcg/day).
- Weeks 13-26 (2nd trimester) → iron supplementation initiation.
- Weeks 27+ (3rd trimester) → continued iron + calcium emphasis.
"""


def _build_user_message(query: str, profile: UserProfile | None) -> str:
    """
    Format the profile + query for the HyDE prompt.

    When profile is None, HyDE still runs but with a generic prompt — the
    hypothetical answer will be less personalised but still closer to
    answer-style prose than the raw question is. Useful for callers that
    don't have profile info (currently unused; pipeline always passes one).
    """
    if profile is None:
        return f"Question: {query}"
    if profile.medical_conditions:
        conditions = ", ".join(c.value for c in profile.medical_conditions)
    else:
        conditions = "None"
    return (
        f"User profile:\n"
        f"- Pregnancy week: {profile.pregnancy_week}\n"
        f"- Diet type: {profile.diet_type.value}\n"
        f"- Medical conditions: {conditions}\n"
        f"\n"
        f"Question: {query}"
    )


@observe(name="hyde_generate")
def generate_hypothetical_answer(query: str, profile: UserProfile | None = None) -> str:
    """
    Generate a hypothetical answer to the query, personalised by profile.

    The returned string is meant to be embedded as the retrieval query —
    NOT shown to the user. It will contain plausible-but-uncited claims
    by design (the user's actual answer goes through the production
    pipeline with grounded retrieval + the strict answer-LLM prompt).

    Traced as its own Langfuse span so we can see (a) latency added by
    HyDE, (b) the actual hypothetical answer when debugging surprising
    retrieval results, and (c) the prompt version for reproducibility.
    """
    user_message = _build_user_message(query, profile)
    update_current_span(
        input={
            "query": query,
            "profile_summary": (
                f"week={profile.pregnancy_week}, diet={profile.diet_type.value}, "
                f"conditions={[c.value for c in profile.medical_conditions]}"
            ) if profile else "(no profile)",
            "prompt_version": PROMPT_VERSION,
        },
    )

    response = get_openai_client().chat.completions.create(
        model=settings.hyde_model,
        temperature=settings.hyde_temperature,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    hypothetical = response.choices[0].message.content.strip()
    update_current_span(output=hypothetical)
    return hypothetical
