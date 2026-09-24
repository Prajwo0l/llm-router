"""
Aggregates benchmark/run_benchmark.py's raw_results.jsonl into the numbers
the project brief asked for: cost, p50/p95 latency, cache-hit-rate, and
quality-vs-baseline, per arm, plus a failure analysis. Prints a markdown
report to stdout and writes it to benchmark/results/report.md.

Quality-vs-baseline is computed only over MMLU + GSM8K rows (the only ones
with an objective graded_correct), reported as router accuracy minus
strongest-arm accuracy -- a negative number means the router traded some
accuracy for cost savings, which is the tradeoff this whole project is
about surfacing, not hiding.

This script does not judge whether the resulting numbers are "good" --
that's explicitly left for a human to look at together (per the project's
own instruction: benchmark scores get reviewed, not auto-approved).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ArmStats:
    arm: str
    n_requests: int
    n_success: int
    n_failed: int
    total_cost_usd: float
    mean_cost_usd: float
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    cache_hit_rate: float
    graded_n: int
    accuracy: float | None
    failures_by_source: dict[str, int]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def load_results(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_arm_stats(arm: str, rows: list[dict]) -> ArmStats:
    arm_rows = [r for r in rows if r["arm"] == arm]
    successes = [r for r in arm_rows if r["success"]]
    failures = [r for r in arm_rows if not r["success"]]

    costs = [r["estimated_cost_usd"] for r in successes if r["estimated_cost_usd"] is not None]
    latencies = [r["latency_ms"] for r in successes if r["latency_ms"] is not None]
    cache_hits = [r for r in successes if r.get("cache_hit")]

    graded = [r for r in successes if r["graded_correct"] is not None]
    accuracy = (sum(1 for r in graded if r["graded_correct"]) / len(graded)) if graded else None

    failures_by_source: dict[str, int] = defaultdict(int)
    for r in failures:
        failures_by_source[r["source"]] += 1

    return ArmStats(
        arm=arm,
        n_requests=len(arm_rows),
        n_success=len(successes),
        n_failed=len(failures),
        total_cost_usd=sum(costs),
        mean_cost_usd=statistics.mean(costs) if costs else 0.0,
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        cache_hit_rate=(len(cache_hits) / len(successes)) if successes else 0.0,
        graded_n=len(graded),
        accuracy=accuracy,
        failures_by_source=dict(failures_by_source),
    )


def render_report(all_stats: list[ArmStats], rows: list[dict]) -> str:
    lines = ["# Benchmark report", ""]

    lines.append("## Summary")
    lines.append("")
    lines.append("| Arm | Requests | Success | Total cost (USD) | Mean cost/req | p50 latency (ms) | p95 latency (ms) | Cache hit rate | Accuracy (MMLU+GSM8K) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in all_stats:
        acc = f"{s.accuracy:.1%}" if s.accuracy is not None else "n/a"
        p50 = f"{s.p50_latency_ms:.0f}" if s.p50_latency_ms is not None else "n/a"
        p95 = f"{s.p95_latency_ms:.0f}" if s.p95_latency_ms is not None else "n/a"
        lines.append(
            f"| {s.arm} | {s.n_requests} | {s.n_success} | ${s.total_cost_usd:.4f} | "
            f"${s.mean_cost_usd:.6f} | {p50} | {p95} | {s.cache_hit_rate:.1%} | {acc} |"
        )

    router_stats = next((s for s in all_stats if s.arm == "router"), None)
    strongest_stats = next((s for s in all_stats if s.arm == "strongest"), None)
    cheapest_stats = next((s for s in all_stats if s.arm == "cheapest"), None)

    if router_stats and strongest_stats:
        lines.append("")
        lines.append("## Router vs always-strongest")
        lines.append("")
        cost_savings_pct = (
            (1 - router_stats.total_cost_usd / strongest_stats.total_cost_usd) * 100
            if strongest_stats.total_cost_usd else 0.0
        )
        acc_delta = (
            (router_stats.accuracy - strongest_stats.accuracy)
            if router_stats.accuracy is not None and strongest_stats.accuracy is not None else None
        )
        lines.append(f"- Cost savings vs always-strongest: **{cost_savings_pct:.1f}%**")
        if acc_delta is not None:
            lines.append(f"- Accuracy delta vs always-strongest: **{acc_delta:+.1%}** (negative = router traded accuracy for cost)")

    if router_stats and cheapest_stats:
        lines.append("")
        lines.append("## Router vs always-cheapest")
        lines.append("")
        cost_premium_pct = (
            (router_stats.total_cost_usd / cheapest_stats.total_cost_usd - 1) * 100
            if cheapest_stats.total_cost_usd else 0.0
        )
        acc_delta = (
            (router_stats.accuracy - cheapest_stats.accuracy)
            if router_stats.accuracy is not None and cheapest_stats.accuracy is not None else None
        )
        lines.append(f"- Cost premium vs always-cheapest: **{cost_premium_pct:+.1f}%**")
        if acc_delta is not None:
            lines.append(f"- Accuracy delta vs always-cheapest: **{acc_delta:+.1%}** (positive = router's smarter routing bought back accuracy)")

    lines.append("")
    lines.append("## Failure analysis")
    lines.append("")
    any_failures = any(s.n_failed for s in all_stats)
    if not any_failures:
        lines.append("No failed requests across any arm.")
    else:
        for s in all_stats:
            if not s.n_failed:
                continue
            lines.append(f"### {s.arm} ({s.n_failed} failed / {s.n_requests})")
            for source, count in s.failures_by_source.items():
                lines.append(f"- {source}: {count} failures")
            sample_errors = [r["error"] for r in rows if r["arm"] == s.arm and not r["success"]][:5]
            for err in sample_errors:
                lines.append(f"  - `{err}`")

    if router_stats:
        lines.append("")
        lines.append("## Semantic cache: similarity distribution (router arm)")
        lines.append("")
        lines.append(
            "closest_cache_similarity is recorded on every request, hit or miss -- this is what makes "
            "semantic_cache_similarity_threshold tunable from real traffic instead of guesswork."
        )
        lines.append("")
        router_rows_all = [r for r in rows if r["arm"] == "router" and r["success"]]
        hit_sims = [r["closest_cache_similarity"] for r in router_rows_all if r["cache_hit"] and r["closest_cache_similarity"] is not None]
        miss_sims = [r["closest_cache_similarity"] for r in router_rows_all if not r["cache_hit"] and r["closest_cache_similarity"] is not None]

        if not hit_sims and not miss_sims:
            lines.append("No similarity data recorded (cache was empty, disabled, or bypassed for every request).")
        else:
            lines.append("| | n | min | mean | max |")
            lines.append("|---|---|---|---|---|")
            if hit_sims:
                lines.append(f"| Hits | {len(hit_sims)} | {min(hit_sims):.3f} | {statistics.mean(hit_sims):.3f} | {max(hit_sims):.3f} |")
            if miss_sims:
                lines.append(f"| Misses (closest neighbor) | {len(miss_sims)} | {min(miss_sims):.3f} | {statistics.mean(miss_sims):.3f} | {max(miss_sims):.3f} |")

            near_misses = sorted(miss_sims, reverse=True)[:5]
            if near_misses:
                lines.append("")
                lines.append(
                    f"Closest near-misses (highest similarity among non-hits): {', '.join(f'{s:.3f}' for s in near_misses)}. "
                    "If any of these are prompts you'd actually want treated as a cache hit, that's a signal to "
                    "lower semantic_cache_similarity_threshold -- if none of them should have hit, the threshold "
                    "is doing its job."
                )

    lines.append("")
    lines.append("## Router difficulty-to-tier distribution")
    lines.append("")
    router_rows = [r for r in rows if r["arm"] == "router" and r["success"]]
    tier_counts: dict[str, int] = defaultdict(int)
    for r in router_rows:
        tier_counts[r["tier_used"] or "unknown"] += 1
    lines.append("| Tier | Requests |")
    lines.append("|---|---|")
    for tier, count in sorted(tier_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {tier} | {count} |")

    return "\n".join(lines) + "\n"


def main(input_path: str, output_path: str) -> None:
    import os

    rows = load_results(input_path)
    arms_present = sorted({r["arm"] for r in rows}, key=lambda a: ["strongest", "cheapest", "router"].index(a) if a in ["strongest", "cheapest", "router"] else 99)
    all_stats = [compute_arm_stats(arm, rows) for arm in arms_present]

    report_text = render_report(all_stats, rows)
    print(report_text)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default="benchmark/results/raw_results.jsonl")
    parser.add_argument("--output", type=str, default="benchmark/results/report.md")
    args = parser.parse_args()
    main(input_path=args.input, output_path=args.output)
