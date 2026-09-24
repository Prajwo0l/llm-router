"""
Builds the labeled training set for the difficulty classifier.

This script is meant to run in Colab (see training/README.md), NOT on this
machine -- it downloads MMLU + GSM8K via the `datasets` library, which is
exactly the kind of local heavy-dependency install this project is
avoiding. It's written and reviewed here; it is not run here.

Labeling strategy (a placeholder, and deliberately a simple one -- see the
"better alternatives" note below):
    - GSM8K (multi-step grade-school math word problems) -> "hard"
    - MMLU (single-shot knowledge/reasoning recall across 57 subjects) -> "medium"
    - A small hand-written set of short factual/greeting/definition prompts -> "easy"

This is a dataset-source-driven heuristic, not a per-example difficulty
measurement. It's a defensible starting point (GSM8K genuinely requires
more reasoning steps than an MMLU multiple-choice recall question, which
in turn requires more than "what's the capital of France") but it's not
ground truth. A better alternative, if the shipped classifier's accuracy
turns out to matter more than "good enough to beat the heuristic
fallback": label each example by whether a cheap model (gpt-4o-mini)
actually gets it right -- correct-on-first-try -> easy, correct-only-with-
a-strong-model -> hard. That requires burning API calls against a labeling
set, which is why it isn't what's shipped by default; swap in that
approach in Colab if you want it and are fine paying for the labeling
pass.

Output: a single JSONL file, one {"text": ..., "label": ...} per line,
written to training/data/difficulty_dataset.jsonl. train_classifier.py
reads this directly.
"""
from __future__ import annotations

import argparse
import json
import random

EASY_EXAMPLES = [
    "Hi, how are you today?",
    "What's the capital of France?",
    "Say hello in Spanish.",
    "What year did World War II end?",
    "Define the word 'ubiquitous'.",
    "What's 2 + 2?",
    "List three primary colors.",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "Translate 'good morning' into French.",
    "What's the plural of 'cactus'?",
    "Name a mammal that lays eggs.",
    "What does HTTP stand for?",
    "Is the sun a planet or a star?",
    "What's the freezing point of water in Celsius?",
    "Give me a synonym for 'happy'.",
    "What day comes after Monday?",
    "How many continents are there?",
    "What's the opposite of 'up'?",
    "Name the largest ocean on Earth.",
]


def _load_mmlu(sample_size: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(sample_size, len(ds)))
    examples = []
    for i in indices:
        row = ds[i]
        choices_text = "\n".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(row["choices"]))
        text = f"{row['question']}\n{choices_text}"
        examples.append({"text": text, "label": "medium"})
    return examples


def _load_gsm8k(sample_size: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    # "openai/gsm8k", not bare "gsm8k" -- see benchmark/dataset.py's
    # identical note; current `datasets` versions require the full
    # namespace/name repo id.
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(sample_size, len(ds)))
    return [{"text": ds[i]["question"], "label": "hard"} for i in indices]


def _load_easy(sample_size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = list(EASY_EXAMPLES)
    # Cycle with light reuse if sample_size exceeds the hand-written pool --
    # fine for a placeholder dataset; replace EASY_EXAMPLES with a larger
    # real source before relying on this for a production classifier.
    examples = []
    while len(examples) < sample_size:
        rng.shuffle(pool)
        examples.extend({"text": t, "label": "easy"} for t in pool)
    return examples[:sample_size]


def build_dataset(per_class: int, seed: int, output_path: str) -> None:
    import os

    examples = _load_easy(per_class, seed) + _load_mmlu(per_class, seed) + _load_gsm8k(per_class, seed)
    random.Random(seed).shuffle(examples)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples ({per_class} per class) to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=500, help="Number of examples per difficulty label.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="training/data/difficulty_dataset.jsonl")
    args = parser.parse_args()
    build_dataset(per_class=args.per_class, seed=args.seed, output_path=args.output)
