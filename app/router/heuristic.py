"""
A non-ML difficulty scorer. This exists so the router is fully functional
(cache, fallback chain, rate limiting, cost tracking, the benchmark harness
-- everything except the *trained* classifier) before any fine-tuning has
happened, and remains a documented fallback afterwards if the classifier
checkpoint is ever missing or fails to load.

This is deliberately simple and deliberately not claimed to be accurate --
it's a placeholder with legible rules, not a model. The benchmark's
"always cheapest / always strongest / router" comparison is exactly what
tells you whether it's good enough to ship on its own, or whether the
fine-tuned classifier earns its place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_MATH_HINTS = re.compile(r"[=+\-*/^]|\bsolve\b|\bcompute\b|\bintegral\b|\bderivative\b|\bequation\b", re.I)
_CODE_HINTS = re.compile(r"```|\bdef \b|\bclass \b|\bfunction\b|\bimport \b|\bstack trace\b|\btraceback\b", re.I)
_MULTI_STEP_HINTS = re.compile(r"\bstep by step\b|\bfirst.*then\b|\bexplain why\b|\bprove\b|\bcompare and contrast\b", re.I)
_SIMPLE_HINTS = re.compile(r"^\s*(what is|who is|when is|define|translate)\b", re.I)


@dataclass
class DifficultyResult:
    label: str  # "easy" | "medium" | "hard"
    confidence: float  # 0..1 -- this heuristic's confidence is deliberately modest; see note below
    source: str  # "heuristic" | "classifier" -- so callers/logs can tell which path produced this


def score_heuristic(prompt: str) -> DifficultyResult:
    text = prompt.strip()
    length = len(text)
    word_count = len(text.split())

    signals = 0
    if _MATH_HINTS.search(text):
        signals += 1
    if _CODE_HINTS.search(text):
        signals += 1
    if _MULTI_STEP_HINTS.search(text):
        signals += 1
    if word_count > 120:
        signals += 1
    if text.count("?") > 1:
        signals += 1

    if _SIMPLE_HINTS.match(text) and word_count < 20 and signals == 0:
        label, confidence = "easy", 0.65
    elif signals >= 2 or word_count > 250:
        label, confidence = "hard", 0.6
    elif signals == 1 or word_count > 60:
        label, confidence = "medium", 0.55
    else:
        label, confidence = "easy", 0.55

    # Deliberately capped below the classifier's typical confidence range --
    # this is a rules-based guess, not a calibrated model, and the policy
    # layer (policy.py) should be more willing to escalate on a heuristic
    # "medium/hard" call than on a trained classifier's.
    confidence = min(confidence, 0.7)

    return DifficultyResult(label=label, confidence=confidence, source="heuristic")
