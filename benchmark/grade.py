"""
Grades a single response against a benchmark prompt's reference answer.

MMLU and GSM8K have objective answers, so they're graded by extraction +
exact match -- no extra model call, no cost, fully deterministic. Chat
prompts have no single correct answer; two grading modes are offered:

    - "ungraded" (default): chat responses are collected but not scored.
      Quality-vs-baseline for chat is then a human judgment call --
      appropriate here, since the user asked to look at benchmark scores
      together rather than have this script silently invent a quality
      number for free-form text.
    - "llm_judge" (opt-in via --judge flag in run_benchmark.py): asks the
      STRONG tier model to rate the router's chat response against the
      always-strongest arm's response on a 1-5 scale. This costs real API
      calls (one extra STRONG-tier call per chat prompt per arm being
      judged) -- opt-in, not the default, for exactly that reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GradeResult:
    correct: bool | None  # None when ungraded (e.g. chat without --judge)
    score: float | None  # 0.0-1.0, None when ungraded
    detail: str


def grade_mmlu(response_text: str, reference_letter: str) -> GradeResult:
    # Look for a standalone letter A-D, preferring the first one that
    # appears at the start of a line or immediately after "answer is" --
    # models often restate the question before answering.
    patterns = [
        r"answer is[:\s]*\(?([A-D])\)?",
        r"^\(?([A-D])\)?[.\s]",
        r"\b([A-D])\b",
    ]
    found = None
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            found = match.group(1).upper()
            break

    correct = found == reference_letter.upper() if found else False
    return GradeResult(correct=correct, score=1.0 if correct else 0.0, detail=f"extracted={found!r} reference={reference_letter!r}")


def grade_gsm8k(response_text: str, reference_number: str) -> GradeResult:
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", response_text.replace(",", ""))
    try:
        reference_value = float(reference_number)
    except ValueError:
        return GradeResult(correct=None, score=None, detail=f"unparseable reference {reference_number!r}")

    found_value = None
    if numbers:
        try:
            found_value = float(numbers[-1])  # final answer is typically the last number stated
        except ValueError:
            found_value = None

    correct = found_value is not None and abs(found_value - reference_value) < 1e-2
    return GradeResult(
        correct=correct, score=1.0 if correct else 0.0, detail=f"extracted={found_value!r} reference={reference_value!r}"
    )


def grade_chat_ungraded(response_text: str) -> GradeResult:
    return GradeResult(correct=None, score=None, detail="chat prompt, not auto-graded (see grade.py docstring)")


async def grade_chat_llm_judge(prompt: str, candidate_response: str, reference_response: str, judge_complete_fn) -> GradeResult:
    """`judge_complete_fn` is injected (an async callable: str -> str) rather
    than imported here, so this module doesn't need to know about
    providers/router_client.py -- run_benchmark.py wires the actual STRONG-
    tier call in."""
    judge_prompt = (
        "You are grading two AI responses to the same user prompt. Rate the CANDIDATE response's quality "
        "relative to the REFERENCE response on a 1-5 scale (5 = candidate is as good as or better than "
        "reference, 1 = candidate is much worse). Reply with only the digit.\n\n"
        f"USER PROMPT:\n{prompt}\n\nREFERENCE RESPONSE:\n{reference_response}\n\nCANDIDATE RESPONSE:\n{candidate_response}\n\n"
        "Score (1-5):"
    )
    judge_output = await judge_complete_fn(judge_prompt)
    match = re.search(r"[1-5]", judge_output)
    if not match:
        return GradeResult(correct=None, score=None, detail=f"judge returned unparseable output: {judge_output!r}")
    raw_score = int(match.group(0))
    return GradeResult(correct=raw_score >= 4, score=(raw_score - 1) / 4, detail=f"llm_judge_score={raw_score}/5")


def grade(source: str, response_text: str, reference_answer: str | None) -> GradeResult:
    if source == "mmlu" and reference_answer is not None:
        return grade_mmlu(response_text, reference_answer)
    if source == "gsm8k" and reference_answer is not None:
        return grade_gsm8k(response_text, reference_answer)
    return grade_chat_ungraded(response_text)
