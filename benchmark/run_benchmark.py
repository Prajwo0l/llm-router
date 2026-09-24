"""
Runs the 3-arm benchmark against a *running* router API server (talks over
HTTP -- this script never imports app/ directly, so it's exercising the
exact same code path a real caller would hit).

Arms:
    strongest  -- router_force_tier=strong on every request (bypasses
                  difficulty routing entirely; the "always use the best
                  model" baseline).
    cheapest   -- router_force_tier=cheap on every request (the "always use
                  the cheapest model" baseline).
    router     -- model=router:auto, no override -- the actual router
                  making its own difficulty-based tier decisions.

Caching: the strongest/cheapest baseline arms run with
router_bypass_cache=true. Those arms are meant to answer "what would this
cost/take with no smart routing at all" -- letting them piggyback on
cache entries populated by an earlier arm's run would understate their
real cost and isn't what "always call the strongest model" means. Only
the router arm uses the cache, since caching is genuinely part of the
router's design being evaluated.

Usage:
    python benchmark/run_benchmark.py \\
        --base-url http://localhost:8000 \\
        --dataset benchmark/data/benchmark_set.jsonl \\
        --output benchmark/results/raw_results.jsonl \\
        --concurrency 8

Requires the benchmark dataset to already exist (see dataset.py) and the
router API server to already be running -- this script does not start it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass

import httpx

from benchmark.dataset import BenchmarkPrompt, load_benchmark_set
from benchmark.grade import grade

ARMS = ["strongest", "cheapest", "router"]


@dataclass
class RunResult:
    arm: str
    prompt_id: str
    source: str
    category: str
    success: bool
    error: str | None
    latency_ms: float | None
    estimated_cost_usd: float | None
    cache_hit: bool | None
    closest_cache_similarity: float | None
    tier_used: str | None
    difficulty_label: str | None
    response_text: str | None
    graded_correct: bool | None
    graded_score: float | None
    graded_detail: str | None


def _request_payload(arm: str, prompt: BenchmarkPrompt) -> dict:
    payload = {
        "messages": [{"role": "user", "content": prompt.prompt}],
        "temperature": 0.0,  # deterministic-ish grading for MMLU/GSM8K; chat quality isn't temperature-sensitive here
    }
    if arm == "strongest":
        payload["router_force_tier"] = "strong"
        payload["router_bypass_cache"] = True
    elif arm == "cheapest":
        payload["router_force_tier"] = "cheap"
        payload["router_bypass_cache"] = True
    elif arm == "router":
        payload["model"] = "router:auto"
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return payload


async def _run_one(client: httpx.AsyncClient, arm: str, prompt: BenchmarkPrompt, semaphore: asyncio.Semaphore) -> RunResult:
    async with semaphore:
        payload = _request_payload(arm, prompt)
        started = time.monotonic()
        try:
            response = await client.post("/v1/chat/completions", json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return RunResult(
                arm=arm, prompt_id=prompt.id, source=prompt.source, category=prompt.category,
                success=False, error=str(exc), latency_ms=(time.monotonic() - started) * 1000,
                estimated_cost_usd=None, cache_hit=None, closest_cache_similarity=None, tier_used=None,
                difficulty_label=None, response_text=None, graded_correct=None, graded_score=None, graded_detail=None,
            )

        response_text = data["choices"][0]["message"]["content"]
        router_meta = data["router"]
        grade_result = grade(prompt.source, response_text, prompt.reference_answer)

        return RunResult(
            arm=arm, prompt_id=prompt.id, source=prompt.source, category=prompt.category,
            success=True, error=None,
            latency_ms=router_meta["latency_ms"], estimated_cost_usd=router_meta["estimated_cost_usd"],
            cache_hit=router_meta["cache_hit"], closest_cache_similarity=router_meta.get("closest_cache_similarity"),
            tier_used=router_meta["tier_used"],
            difficulty_label=router_meta["difficulty_label"], response_text=response_text,
            graded_correct=grade_result.correct, graded_score=grade_result.score, graded_detail=grade_result.detail,
        )


async def run_benchmark(base_url: str, dataset_path: str, output_path: str, concurrency: int, arms: list[str]) -> None:
    import os

    prompts = load_benchmark_set(dataset_path)
    semaphore = asyncio.Semaphore(concurrency)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    async with httpx.AsyncClient(base_url=base_url) as client:
        with open(output_path, "w") as out_f:
            for arm in arms:
                print(f"--- running arm: {arm} ({len(prompts)} prompts) ---")
                tasks = [_run_one(client, arm, prompt, semaphore) for prompt in prompts]
                completed = 0
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    out_f.write(json.dumps(asdict(result)) + "\n")
                    out_f.flush()
                    completed += 1
                    if completed % 25 == 0:
                        print(f"  {arm}: {completed}/{len(prompts)}")
                print(f"--- arm {arm} done ---")

    print(f"Wrote raw results to {output_path}. Run benchmark/report.py next to aggregate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--dataset", type=str, default="benchmark/data/benchmark_set.jsonl")
    parser.add_argument("--output", type=str, default="benchmark/results/raw_results.jsonl")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--arms", type=str, nargs="+", default=ARMS, choices=ARMS)
    args = parser.parse_args()
    asyncio.run(
        run_benchmark(
            base_url=args.base_url, dataset_path=args.dataset, output_path=args.output,
            concurrency=args.concurrency, arms=args.arms,
        )
    )
