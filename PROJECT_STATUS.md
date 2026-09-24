# llm-router — project notes

## The idea

An eval-driven LLM router: an OpenAI-compatible gateway that sits in front
of multiple model providers and picks the *cheapest* model tier likely to
answer a given prompt well, instead of sending every request to the
strongest (most expensive) model "just in case."

Core pieces of the idea:

- A **difficulty classifier** looks at each incoming prompt and predicts
  how hard it is (easy / medium / hard).
- A **router** maps that prediction to a cost tier (cheap / mid / strong /
  local) and sends the request to the matching model.
- A **semantic cache** (embeddings + similarity search) catches repeated
  or near-duplicate prompts and answers them for free instead of calling a
  model at all.
- If a call fails or the model is uncertain, the router **escalates**
  up the tier ladder rather than just failing or guessing.
- A **benchmark** compares three strategies head-to-head — always use the
  strongest model, always use the cheapest, or use the router — on cost,
  latency, cache-hit-rate, and quality, so the savings claim is measured,
  not assumed.
- Everything ships as **one Docker Compose command**, with rate limiting,
  cost tracking, and optional tracing built in.

## Constraints I was given for this build

- No fine-tuning or heavy ML work on this machine. The classifier's
  training script/notebook is written here but actually *run* in Google
  Colab — this machine never installs torch/transformers/GPU stuff.
- No running/executing anything locally during the build — no `pip
  install`, no starting the server, no running tests. Everything was
  written and manually reviewed, not executed, until you say otherwise.
- Benchmark scores get looked at together once they exist — not something
  the code should assert or auto-judge.

## What's built (all code, all reviewed, none executed yet)

```
app/            FastAPI gateway — routing, providers, cache, rate limiting,
                cost tracking, tracing, dashboard, the /v1/chat/completions endpoint
training/       Difficulty classifier: dataset prep + fine-tuning script +
                a Colab notebook that runs them (Colab-only, not local)
benchmark/      3-arm benchmark: dataset builder, runner, grader, reporter
loadtest/       Locust load test (with-cache / without-cache variants)
docs/           Architecture doc with a diagram and the reasoning behind
                each design decision
tests/          Pure-logic unit tests (no network, no model, no real keys)
docker-compose.yml, Dockerfile, requirements*.txt, .env.example
```

**Dashboard** (`app/dashboard/`) — added after the fact, was in the
original brief but got missed in the first pass. `GET /dashboard`: total
requests/cost, cache hit rate, cost-over-time + tier-distribution charts,
recent-requests table. Static HTML + Chart.js (CDN, no build step),
reading straight from the SQLite cost ledger. Unauthenticated all-keys
view — fine for local use, needs your own auth/proxy rule before it's
public. Empty until real requests flow through the API.

Design choices worth remembering:

- **Heavy dependencies are all optional at import time.** torch/
  transformers (classifier), faiss (cache), OpenTelemetry/LangSmith
  (tracing) are imported lazily. The app boots and serves requests with
  none of them installed — difficulty scoring just falls back to a
  regex/word-count heuristic, the cache disables itself, tracing no-ops.
- **Tiers, not model names.** `app/router/tiers.py` is the one place that
  maps a tier to a real (provider, model, cost) — routing logic never
  hardcodes a model name.
- **Fallback escalates up, not sideways** — a failing cheap-tier call
  retries a couple of times then moves to the next tier up, rather than
  trying another cheap option.
- **Streaming isn't implemented.** `stream: true` returns 501 — wiring
  streaming through retry/fallback/cache is a real design problem that was
  explicitly out of scope for getting the full system running in one pass.

## Budget reality (as of 2026-09-24)

Only an OpenAI key is available, with **$1.98** left on it. Two changes
made to fit that:

- `app/router/tiers.py`: CHEAP/MID/STRONG all temporarily point at
  `gpt-4o-mini` instead of MID/STRONG using Anthropic — with no Anthropic
  key, those tiers would 502 on every "medium"/"hard" prompt otherwise.
  All three tiers cost the same right now, so this tests the
  routing/escalation *logic*, not real quality-tier differences. Swap MID/
  STRONG back once there's an Anthropic key or more budget.
- `docker-compose.yml`: the `ollama` service is now opt-in (Compose
  `profiles: ["local"]`), not started by a plain `docker compose up` —
  one less thing running for a lean first test.
- Corresponding test fixtures (`tests/test_router_client.py`,
  `tests/test_tiers.py`) were updated to not assume a specific provider
  mapping or strictly-increasing tier cost, since that mapping is now
  expected to change based on what keys are available.

## What's left

**1. First real run (nothing has executed yet)**
- `cp .env.example .env`, add `OPENAI_API_KEY` (that's the only key you have)
- `docker compose up --build`
- Hit `/health`, open `/dashboard` (should load empty), then send one
  `/v1/chat/completions` request

**2. Fix whatever breaks on that first run**
Realistic, expected — this is ~4,000 lines written without ever being
executed. Likely spots: a pinned dependency version mismatch, an import
typo, an SDK response-shape assumption that's drifted from what the
current OpenAI SDK actually returns.

**3. Small, cheap smoke test before scaling up anything**
A handful of manual `/v1/chat/completions` calls (a few cents at most on
gpt-4o-mini), watching `/dashboard` and `/v1/usage` to confirm cost
tracking and cache hits behave as expected. Don't run the full benchmark
yet — confirm the pipeline is healthy on pennies first.

**4. A small benchmark pass, sized to the budget**
Instead of the default 300-500 prompts × 3 arms, start with something
like 10 prompts per category (~30 total), router arm only first:
```
python benchmark/dataset.py --mmlu-n 10 --gsm8k-n 10 --chat-n 10
python -m benchmark.run_benchmark --base-url http://localhost:8000 --arms router
python benchmark/report.py
```
Check the actual cost spent (dashboard or `/v1/usage`) before deciding
whether to run the `strongest`/`cheapest` comparison arms too, or scale
the prompt count up.

**5. Train the difficulty classifier in Colab** (optional, do this if step
4 suggests the heuristic scorer isn't good enough on its own)
Open `training/colab_finetune_difficulty_classifier.ipynb`, run it top to
bottom, download the checkpoint, drop it into
`training/checkpoints/difficulty-classifier/`.

**6. Tune based on what the benchmark shows**
- ~~`classifier_confidence_floor`~~ — **done.** The first real benchmark
  run (30 prompts, 2026-09-24) showed 0/30 requests reaching the CHEAP
  tier, because the old default (0.6) sat inside the heuristic scorer's
  actual confidence range (0.55-0.65), causing near-universal escalation.
  Lowered to 0.5 in `app/config.py` and `.env` — re-run the benchmark to
  confirm CHEAP gets used now.
- `semantic_cache_similarity_threshold` (default 0.94) — not yet
  benchmarked against real cache-hit-rate data (the 30-prompt run had 0%
  hits by design, no repeated prompts in the set).
- Once there's more budget or an Anthropic key: restore MID/STRONG to
  real, distinct models in `app/router/tiers.py` and re-run the benchmark
  — the current OpenAI-only numbers (including the 95% accuracy result)
  reflect gpt-4o-mini's own quality on this sample, not a real tier
  quality/cost tradeoff, since all three tiers are currently the same model.

**7. Optional / later**
- Load test with Locust once the server is stable under normal traffic
- Start the `ollama` service (`docker compose --profile local up`) and
  pull a model if the LOCAL tier should be live
- Swap in the LLM-judge grading mode for chat prompts in the benchmark
  (off by default — costs extra API calls)
