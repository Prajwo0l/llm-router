# llm-router

An eval-driven LLM router: an OpenAI-compatible gateway that picks the
cheapest model tier likely to answer a given prompt well, caches
semantically-similar repeat requests, and falls back up the cost/quality
ladder on failure -- instead of every request paying for the strongest
model "just in case."

See [`docs/architecture.md`](docs/architecture.md) for the full request-flow
diagram and the reasoning behind each design decision.

## Status

This is a complete, runnable system with two things still outstanding by
design, not by accident:

1. **The difficulty classifier isn't trained yet.** The router works today
   using a rules-based heuristic scorer (`app/router/heuristic.py`) --
   fine-tuning a real classifier happens in Google Colab (`training/`), not
   on whatever machine runs the API. Nothing here requires a GPU or a local
   ML install to work end-to-end.
2. **Benchmark numbers haven't been generated/reviewed yet.** The harness
   (`benchmark/`) is complete and ready to run against a live server; the
   actual cost/latency/quality numbers are something to look at together
   once it's been run, not something this codebase asserts.
3. **`app/router/tiers.py` currently points every paid tier (CHEAP/MID/
   STRONG) at `gpt-4o-mini`**, a temporary state for testing against a
   single, budget-constrained OpenAI key -- see that file's comment.
   Escalation logic is fully testable this way even though all three tiers
   cost the same right now; swap MID/STRONG back to real distinct models
   (Anthropic, or a pricier OpenAI model) once you have the budget/keys for
   it.

## Quickstart

```bash
cp .env.example .env
# fill in OPENAI_API_KEY in .env -- with no key at all, the server still
# boots and /health works, but every chat completion returns a clean 502
# ("all tiers exhausted") since there's no provider to send it to

docker compose up --build
```

That's the whole "one command" story: just the `router` service builds and
starts the API on `:8000`. The `ollama` service (for the LOCAL tier) is
opt-in and NOT started by a plain `docker compose up` -- see the comment
in `docker-compose.yml` if you want it running alongside the router.

Then:

```bash
curl http://localhost:8000/health

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

Any OpenAI-client library works unmodified against this -- just point
`base_url` at `http://localhost:8000/v1`.

Open `http://localhost:8000/dashboard` for a live view of routing
decisions, cost over time, tier distribution, and cache hit rate (see
"Dashboard" below).

## Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-cache.txt
cp .env.example .env  # fill in keys
uvicorn app.main:app --reload
```

## Project layout

```
app/                 The FastAPI service
  config.py           Settings (env-driven, pydantic-settings)
  models.py            OpenAI-compatible request/response schemas
  main.py               The /v1/chat/completions endpoint, wiring everything together
  router/
    heuristic.py          Rules-based difficulty scorer (the default, until trained)
    classifier.py          Wraps a trained checkpoint, falls back to heuristic
    policy.py               Difficulty -> tier decision, confidence-gated escalation
    tiers.py                  Tier <-> (provider, model, cost) registry
  providers/           OpenAI / Anthropic / Ollama backends + fallback orchestration
  cache/               Semantic cache (embeddings + FAISS)
  middleware/           Per-key rate limiting, SQLite cost ledger
  telemetry/             OpenTelemetry / LangSmith tracing (both optional, off by default)
  dashboard/             Minimal live metrics dashboard (static HTML + Chart.js, served by FastAPI)

training/             Difficulty classifier fine-tuning -- runs in Colab, not locally
benchmark/             3-arm (always-strongest / always-cheapest / router) benchmark harness
loadtest/               Locust load test, with/without cache
docs/                   Architecture doc + diagram
tests/                  Unit tests -- pure logic, no network/model calls
```

## Running the benchmark

```bash
pip install -r requirements-benchmark.txt

# 1. build the fixed prompt set (MMLU + GSM8K + chat, ~300-400 prompts by default)
python benchmark/dataset.py --output benchmark/data/benchmark_set.jsonl

# 2. with the server already running (docker compose up, or uvicorn locally)
# -- run as a module (-m), not as a script path, since it imports from the
# benchmark package (dataset.py, grade.py) and needs the project root on
# the Python path for that to resolve:
python -m benchmark.run_benchmark --base-url http://localhost:8000

# 3. aggregate into a report
python benchmark/report.py
```

Produces `benchmark/results/report.md`: cost, p50/p95 latency, cache-hit
rate, and accuracy-vs-baseline for all three arms, plus a failure
breakdown. See `benchmark/grade.py` and `benchmark/report.py`'s docstrings
for exactly how quality is scored and what's deliberately left ungraded.

## Dashboard

`GET /dashboard` -- total requests, total cost, cache hit rate, cost-over-
time and requests-by-tier charts, and a table of recent requests, reading
straight from the SQLite cost ledger (`app/middleware/cost_tracking.py`).
No build step: one static HTML page, vanilla JS, Chart.js from a CDN,
polling `GET /dashboard/api/summary` every 30s.

It's an unauthenticated, all-keys aggregate view -- fine for local/
single-operator use, but put it behind your own auth or a reverse-proxy
rule before exposing this port publicly (see `app/dashboard/routes.py`'s
docstring). There's nothing to look at until requests actually flow
through `/v1/chat/completions` -- the benchmark run in the next section is
a good way to populate it.

## Load testing

```bash
pip install locust

# without cache (every prompt unique -- measures raw router+provider latency)
locust -f loadtest/locustfile.py --host http://localhost:8000 --tags nocache \
  -u 50 -r 5 -t 3m --headless --csv=loadtest/results/nocache

# with cache (small repeated prompt pool -- measures effect of cache hits)
locust -f loadtest/locustfile.py --host http://localhost:8000 --tags cached \
  -u 50 -r 5 -t 3m --headless --csv=loadtest/results/cached
```

## Training the difficulty classifier

Entirely in Colab -- see [`training/README.md`](training/README.md) and
`training/colab_finetune_difficulty_classifier.ipynb`. Copy the resulting
checkpoint back into `training/checkpoints/difficulty-classifier/` and the
running API picks it up automatically (no code change, no restart-specific
step beyond the next request triggering a lazy load).

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

All tests are pure-logic (heuristic scoring, tier/policy decisions, rate
limiter math, grading regexes, fallback orchestration against fake
providers) -- no network calls, no real API keys, no model downloads.

## Configuration

See `.env.example` for every setting, or `app/config.py` for defaults and
what each one controls. Every provider API key is optional -- a provider
with no key configured is simply skipped by the fallback chain rather than
causing a startup error, so this runs fine with just one provider key set.
