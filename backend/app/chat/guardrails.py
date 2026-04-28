"""
guardrails.py — Canned responses for non-answer pipeline outcomes.

Three fixed strings that the pipeline returns when it short-circuits:

  EMERGENCY_RESPONSE     — classifier flagged urgent medical situation
  OUT_OF_SCOPE_RESPONSE  — classifier flagged unrelated topic
  NO_RESULTS_RESPONSE    — retrieval found nothing above similarity_threshold

Why hard-coded (no LLM call):
  These paths exist precisely to skip the LLM. Generating them dynamically
  would add latency and a small failure surface for messages where we have
  zero appetite for variability — especially the emergency redirect.

Why this file is just constants:
  The triage logic that USED to live here (keyword regex matching) moved
  to `classifier.py`. This module is now a tiny string registry — but a
  registry, not free-floating constants in pipeline.py, so the eval suite
  and any future channel (SMS, voice) can import the same canonical text.
"""

EMERGENCY_RESPONSE = (
    "It sounds like you may be experiencing something urgent. "
    "Please contact a qualified healthcare provider or go to your nearest hospital immediately. "
    "In India, you can call 108 for emergency medical services."
)

OUT_OF_SCOPE_RESPONSE = (
    "I'm Poshan Saathi, a nutrition companion for pregnant women. "
    "I can only help with questions about nutrition and antenatal care."
)

NO_RESULTS_RESPONSE = (
    "I'm Poshan Saathi, a nutrition companion for pregnant women. "
    "I don't have information about that topic in my reference guidelines."
)
