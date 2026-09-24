"""
Builds the fixed benchmark prompt set: MMLU + GSM8K + a small hand-written
chat set, each with a known correct answer where one exists (multiple-choice
letter for MMLU, numeric final answer for GSM8K) so grade.py can score
router output without another model call for MMLU/GSM8K specifically. Chat
prompts have no single correct answer and are graded via similarity-to-
reference (see grade.py) or left for manual review.

Unlike training/prepare_dataset.py, this pulls a much smaller, *fixed*
sample (a fixed seed -- same 300-500 prompts every run) so benchmark runs
are comparable across code changes and across the three arms (always-
strongest / always-cheapest / router). Uses the `datasets` library, which
is a real dependency for anyone running the benchmark -- but note the
benchmark is something you run when you're ready to look at numbers, not
part of the always-importable app/ package, so this is fine to keep as a
top-level import here.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass


@dataclass
class BenchmarkPrompt:
    id: str
    source: str  # "mmlu" | "gsm8k" | "chat"
    category: str
    prompt: str
    reference_answer: str | None  # letter for MMLU, number for GSM8K, None for chat


CHAT_PROMPTS = [
    ("Explain what a REST API is in two sentences.", "general"),
    ("Write a short, friendly out-of-office email.", "writing"),
    ("What's a good icebreaker question for a team meeting?", "general"),
    ("Summarize the plot of Romeo and Juliet in one paragraph.", "writing"),
    ("Give me three tips for staying focused while working from home.", "advice"),
    ("What's the difference between a list and a tuple in Python?", "technical"),
    ("Write a haiku about autumn.", "creative"),
    ("How do I politely decline a meeting invite?", "advice"),
    ("What are some good beginner house plants?", "general"),
    ("Explain the difference between HTTP and HTTPS.", "technical"),
    ("Draft a two-sentence product description for a reusable water bottle.", "writing"),
    ("What questions should I ask before signing an apartment lease?", "advice"),
    ("Give an analogy that explains how a cache works.", "technical"),
    ("Write a short congratulatory message for a coworker's promotion.", "writing"),
    ("What's the difference between weather and climate?", "general"),
    ("Suggest a simple recipe using only eggs, bread, and cheese.", "general"),
    ("Explain what 'technical debt' means to a non-engineer.", "technical"),
    ("What's a polite way to ask a friend to repay a small loan?", "advice"),
    ("Write a one-sentence tagline for a coffee shop.", "creative"),
    ("What's the difference between a resume and a CV?", "general"),
]


def _load_mmlu(n: int, seed: int) -> list[BenchmarkPrompt]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in indices:
        row = ds[i]
        choices_text = "\n".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(row["choices"]))
        prompt = (
            f"{row['question']}\n{choices_text}\n\n"
            "Answer with just the letter of the correct choice."
        )
        answer_letter = chr(65 + int(row["answer"]))
        out.append(
            BenchmarkPrompt(
                id=f"mmlu-{i}", source="mmlu", category=row.get("subject", "unknown"),
                prompt=prompt, reference_answer=answer_letter,
            )
        )
    return out


def _load_gsm8k(n: int, seed: int) -> list[BenchmarkPrompt]:
    from datasets import load_dataset

    # "openai/gsm8k", not bare "gsm8k" -- current `datasets` versions
    # require the full namespace/name repo id; the old bare shortname
    # this project previously used no longer resolves.
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in indices:
        row = ds[i]
        # GSM8K answers end with "#### <number>"
        final_answer = row["answer"].split("####")[-1].strip().replace(",", "")
        prompt = f"{row['question']}\n\nShow your work, then give the final numeric answer on its own line."
        out.append(
            BenchmarkPrompt(id=f"gsm8k-{i}", source="gsm8k", category="math", prompt=prompt, reference_answer=final_answer)
        )
    return out


def _load_chat(n: int, seed: int) -> list[BenchmarkPrompt]:
    rng = random.Random(seed)
    pool = list(CHAT_PROMPTS)
    rng.shuffle(pool)
    selected = (pool * (n // len(pool) + 1))[:n]
    return [
        BenchmarkPrompt(id=f"chat-{i}", source="chat", category=cat, prompt=text, reference_answer=None)
        for i, (text, cat) in enumerate(selected)
    ]


def build_benchmark_set(mmlu_n: int, gsm8k_n: int, chat_n: int, seed: int, output_path: str) -> None:
    import os

    prompts = _load_mmlu(mmlu_n, seed) + _load_gsm8k(gsm8k_n, seed) + _load_chat(chat_n, seed)
    random.Random(seed).shuffle(prompts)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(asdict(p)) + "\n")

    print(f"Wrote {len(prompts)} benchmark prompts ({mmlu_n} MMLU, {gsm8k_n} GSM8K, {chat_n} chat) to {output_path}")


def load_benchmark_set(path: str) -> list[BenchmarkPrompt]:
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append(BenchmarkPrompt(**row))
    return prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmlu-n", type=int, default=150)
    parser.add_argument("--gsm8k-n", type=int, default=150)
    parser.add_argument("--chat-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=str, default="benchmark/data/benchmark_set.jsonl")
    args = parser.parse_args()
    build_benchmark_set(
        mmlu_n=args.mmlu_n, gsm8k_n=args.gsm8k_n, chat_n=args.chat_n, seed=args.seed, output_path=args.output
    )
