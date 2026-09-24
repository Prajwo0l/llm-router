# llm-router — what this is, how it works, why it matters

## The pitch, in one paragraph

Every LLM app makes the same mistake: it sends every request to the same
model. A one-word factual question and a multi-step reasoning problem get
routed identically, usually to whichever model is "good enough" for the
hardest case — which means you're paying premium prices for the 80% of
traffic that didn't need it. **llm-router** is a gateway that sits between
your app and your model providers, looks at each incoming prompt, decides
how hard it actually is, and sends it to the cheapest model likely to
answer it well. Repeated or near-duplicate questions get answered from a
cache instead of hitting a model at all. If a cheap model fails or seems
unreliable on a given request, the router automatically escalates to a
stronger one rather than just erroring out. And instead of asserting any
of this saves money, it ships with a benchmark that measures it.

## The problem it's solving

Teams building on LLM APIs face a real tension:

- Use the strongest model everywhere → correct and reliable, but the bill
  scales with volume regardless of how simple most requests are.
- Use the cheapest model everywhere → cheap, but quality degrades on the
  requests that actually needed a stronger model, and you find out from
  angry users, not metrics.
- Hand-write routing rules per use case → works until your product
  surface grows past the handful of cases you thought to write rules for.

The router's answer: make the routing decision automatically, per
request, based on a measurement of difficulty — and make the tradeoff
*visible* (via the dashboard and the benchmark) instead of hoping it's
working.

## How we approached it

### 1. Build the whole system before the "smart" part exists

The hardest part of this idea — a fine-tuned classifier that predicts
prompt difficulty — takes a training run, a GPU, and a labeled dataset.
None of that needed to block building everything *around* it. So the
difficulty scorer was built as a swappable interface from day one: a
simple, transparent rules-based heuristic (regex signals for math/code/
multi-step reasoning, word-count thresholds) stands in until a trained
classifier exists, and the router, cache, fallback logic, cost tracking,
and dashboard were all built and wired up against that heuristic. The
trained classifier is a drop-in replacement, not a prerequisite — you copy
a checkpoint into a folder and the system starts using it automatically,
no code change.

This is also why nothing here required a GPU or heavy ML install on the
machine building it: every ML-heavy dependency (torch/transformers for the
classifier, faiss for the cache, tracing SDKs) is imported *lazily*, only
when actually used, and the system degrades gracefully — not crashes — if
that dependency isn't installed. You can run the full API with zero ML
packages present.

### 2. Route by tier, not by model name

Rather than hardcoding "send easy prompts to gpt-4o-mini," the system
routes to a named tier (`cheap` / `mid` / `strong` / `local`), and a
single registry file maps each tier to an actual (provider, model, cost)
triple. Swapping which model backs "strong" — say, a new release comes out
— is a one-line change in one file, not a search-and-replace through
routing logic scattered across the codebase.

### 3. Treat failure as routine, not exceptional

API calls fail: rate limits, timeouts, outages. The router treats a
failure at one tier as a signal to retry briefly, then escalate to the
next tier up — the assumption being that if the cheap model is having
problems, the goal is "still get a good answer," not "find another cheap
option." The same escalation logic also kicks in when the difficulty
scorer itself is *uncertain* about a prediction — a low-confidence "easy"
guess gets bumped up a tier rather than trusted blindly, because the cost
of over-escalating occasionally is much smaller than the cost of a
confidently-wrong routing decision on a request that mattered.

### 4. Cache before you route

A semantic cache (prompt → embedding → similarity search) sits in front of
the routing decision entirely. If a near-duplicate of a previous prompt
comes in, it's answered instantly from the cache — no difficulty scoring,
no model call, no cost. This is often the single biggest cost lever for
any product with repeated questions (FAQs, support tickets, retries), and
it's evaluated on its own axis (cache-hit-rate) in the benchmark rather
than assumed to help.

### 5. Measure the claim, don't assert it

The whole premise — "smart routing saves money without losing much
quality" — is only worth something if it's actually measured. The
benchmark harness runs the same fixed prompt set (MMLU for knowledge/
reasoning, GSM8K for multi-step math, a hand-written chat set) through
three arms: always use the strongest model, always use the cheapest, and
let the router decide. It reports cost, p50/p95 latency, cache-hit-rate,
and accuracy for each arm side by side, plus a failure breakdown — so the
tradeoff is a number you can look at, not a claim you have to trust.

### 6. Make the system observable, not a black box

A minimal live dashboard reads the same cost ledger every request writes
to, and shows routing decisions, spend, cache performance, and recent
requests in real time. The point isn't a polished product UI — it's that
"is this actually working as intended" should be answerable by looking at
a page, not by reading logs.

## What this approach actually buys you

- **Lower cost at the same quality bar** (the thing the benchmark exists
  to prove, not assert) — most traffic in most products is not the hard
  case, and paying strongest-model prices for all of it is waste.
- **Resilience for free.** Provider outages, rate limits, and transient
  failures are handled by the same escalation logic that handles
  difficulty — there's no separate "what if OpenAI is down" code path
  bolted on afterward.
- **No vendor lock-in.** OpenAI, Anthropic, and a local Ollama model are
  all just entries in the tier registry behind the same interface. Adding
  a fourth provider is implementing one small interface, not touching
  routing logic.
- **A system that's honest about what it doesn't know yet.** The
  heuristic scorer isn't dressed up as more accurate than it is: its own
  docstring says outright that the benchmark is what determines whether
  it's good enough to keep, or whether the trained classifier earns its
  place. That's a deliberate property of how this was built, not a gap.
- **It's usable today, with zero training.** You don't have to wait for a
  Colab run to get value — the heuristic-only router is a complete,
  working system on its own; the classifier is an upgrade path, not a
  blocker.

## What makes this more than an API wrapper

A thin wrapper around a few chat completion calls wouldn't need: a
provider abstraction with a typed retryable/non-retryable error boundary,
confidence-gated escalation policy, a cache layer that's architecturally
separate from the routing decision, a cost ledger schema designed to
answer real questions (spend by tier, spend by key, time series), a
benchmark methodology with three comparison arms and an explicit note on
what's *not* auto-graded and why (chat quality has no single right
answer), or a dashboard reading live off that ledger. Those pieces exist
because the goal was a system whose behavior can be reasoned about,
measured, and trusted — not just a demo that happens to work on the happy
path.

## Where it stands right now

Fully written, fully reviewed, **not yet run** — every module was built to
degrade gracefully and was traced through by hand rather than executed,
per how this build was scoped. See [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
for the concrete list of what's left (first real run, fixing whatever
breaks, training the classifier in Colab, running the benchmark, tuning
based on the results) and [`docs/architecture.md`](docs/architecture.md)
for the full request-flow diagram and the reasoning behind each
lower-level design decision.
