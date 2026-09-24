"""Pure-logic tests for the heuristic difficulty scorer -- no network, no
model, no fixtures beyond plain strings. Expectations here are derived
directly from the signal rules in app/router/heuristic.py, not guessed."""
from app.router.heuristic import score_heuristic


def test_simple_question_pattern_scores_easy_with_higher_confidence():
    result = score_heuristic("What is the capital of France?")
    assert result.label == "easy"
    assert result.confidence == 0.65


def test_plain_greeting_scores_easy():
    result = score_heuristic("Hi, how are you?")
    assert result.label == "easy"


def test_math_plus_multistep_signals_score_hard():
    result = score_heuristic("Solve for x: 3x^2 + 5x - 2 = 0, show your work step by step.")
    assert result.label == "hard"


def test_single_signal_scores_medium():
    # One math signal ("equation"), no others, short enough to stay under
    # the word-count thresholds -- exactly one signal maps to "medium".
    result = score_heuristic("Can you help me understand this equation a bit better please?")
    assert result.label == "medium"


def test_long_prompt_without_signals_scores_hard_on_length_alone():
    long_prompt = " ".join(["word"] * 260)
    result = score_heuristic(long_prompt)
    assert result.label == "hard"


def test_source_is_always_heuristic():
    result = score_heuristic("What's the weather like?")
    assert result.source == "heuristic"


def test_confidence_never_exceeds_cap():
    prompts = [
        "Hi, how are you?",
        "What is the capital of France?",
        "Solve for x: 3x^2 + 5x - 2 = 0, show your work step by step.",
        " ".join(["word"] * 300),
    ]
    for prompt in prompts:
        assert score_heuristic(prompt).confidence <= 0.7


def test_empty_prompt_does_not_crash():
    result = score_heuristic("")
    assert result.label in {"easy", "medium", "hard"}
