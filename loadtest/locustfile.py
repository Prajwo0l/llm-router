"""
Locust load test for the router API. Two task sets, selectable via
`--tags`, so you can run the "with cache" and "without cache" comparisons
the project brief asks for as two separate runs against the same server:

    # without cache: every request is a distinct prompt, cache never hits
    locust -f loadtest/locustfile.py --host http://localhost:8000 --tags nocache

    # with cache: requests are drawn from a small fixed pool of prompts,
    # so after warmup most requests should be cache hits
    locust -f loadtest/locustfile.py --host http://localhost:8000 --tags cached

Run headless with e.g. `-u 50 -r 5 -t 3m --headless --csv=loadtest/results/run1`
to get CSV output Locust can also plot from.

Each simulated user gets its own fake API key (see RouterUser.on_start) so
-u N actually represents N separate clients, each with their own rate-limit
bucket -- without this, every user shares the "anonymous" bucket and the
test mostly measures the rate limiter rejecting a single hammering client,
not realistic multi-tenant load.
"""
from __future__ import annotations

import random
import uuid

from locust import HttpUser, between, tag, task

# A small fixed pool -- reused across requests within a run so the "cached"
# task set actually has repeats to hit against. Deliberately short/simple
# prompts; this is a load test, not a quality eval (that's benchmark/).
_CACHEABLE_PROMPTS = [
    "What's the capital of Japan?",
    "Summarize the water cycle in two sentences.",
    "What's 15% of 200?",
    "Name three benefits of regular exercise.",
    "What's the difference between TCP and UDP?",
    "Write a one-sentence definition of machine learning.",
    "What year did the Berlin Wall fall?",
    "Give me a synonym for 'efficient'.",
]


class RouterUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        # Each simulated user gets its own fake API key, so Locust's -u
        # users actually exercise the rate limiter's per-key isolation
        # instead of all sharing the single "anonymous" bucket (see
        # app/main.py's _resolve_api_key -- no Authorization header means
        # every request falls back to the same shared key). Without this,
        # a run with -u 20 measures one client hammering the server 20x
        # concurrently and hitting default_rate_limit_per_minute almost
        # immediately, not 20 separate clients under normal load.
        self.client.headers["Authorization"] = f"Bearer loadtest-user-{uuid.uuid4().hex[:12]}"

    @tag("nocache")
    @task(3)
    def chat_unique_prompt(self):
        # A UUID in every prompt guarantees a cache miss -- this measures
        # raw router+provider latency with the cache providing no help,
        # the "without cache" arm of the load test.
        prompt = f"Briefly explain the concept of {uuid.uuid4().hex[:8]} in your own words."
        self._post_chat(prompt, name="/v1/chat/completions [nocache]")

    @tag("cached")
    @task(3)
    def chat_repeated_prompt(self):
        prompt = random.choice(_CACHEABLE_PROMPTS)
        self._post_chat(prompt, name="/v1/chat/completions [cacheable]")

    @tag("nocache")
    @tag("cached")
    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")

    def _post_chat(self, prompt: str, name: str) -> None:
        self.client.post(
            "/v1/chat/completions",
            json={"model": "router:auto", "messages": [{"role": "user", "content": prompt}]},
            name=name,
        )
