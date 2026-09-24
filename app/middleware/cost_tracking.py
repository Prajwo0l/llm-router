"""
Cost ledger: every completion (cache hit or real model call) gets one row.
Backed by sqlite3 (stdlib, always available -- not a "heavy" dependency
like torch/faiss, so this is a normal top-level import) rather than an
external database, matching the one-command-docker-compose goal: nothing
extra to stand up just to track spend.

Reads/writes are synchronous sqlite3 calls run in a thread via
asyncio.to_thread, since sqlite3's connection objects aren't safe to share
across event-loop tasks without care, and a single writer thread avoids
"database is locked" errors under concurrent requests.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from dataclasses import dataclass

from app.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cost_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    api_key TEXT NOT NULL,
    tier TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    estimated_cost_usd REAL NOT NULL,
    cache_hit INTEGER NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cost_events_api_key ON cost_events(api_key);
CREATE INDEX IF NOT EXISTS idx_cost_events_ts ON cost_events(ts);
"""


@dataclass
class CostEvent:
    api_key: str
    tier: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    cache_hit: bool
    latency_ms: float
    ts: float = 0.0

    def __post_init__(self):
        if not self.ts:
            self.ts = time.time()


class CostLedger:
    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _record_sync(self, event: CostEvent) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO cost_events "
                "(ts, api_key, tier, provider, model, prompt_tokens, completion_tokens, "
                " estimated_cost_usd, cache_hit, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.ts,
                    event.api_key,
                    event.tier,
                    event.provider,
                    event.model,
                    event.prompt_tokens,
                    event.completion_tokens,
                    event.estimated_cost_usd,
                    int(event.cache_hit),
                    event.latency_ms,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    async def record(self, event: CostEvent) -> None:
        await asyncio.to_thread(self._record_sync, event)

    def _summary_sync(self, api_key: str | None, since_ts: float | None) -> dict:
        conn = sqlite3.connect(self._db_path)
        try:
            conditions = []
            params: list = []
            if api_key is not None:
                conditions.append("api_key = ?")
                params.append(api_key)
            if since_ts is not None:
                conditions.append("ts >= ?")
                params.append(since_ts)
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            row = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(estimated_cost_usd), 0), "
                f"COALESCE(SUM(cache_hit), 0) FROM cost_events {where}",
                params,
            ).fetchone()
            total_requests, total_cost, cache_hits = row

            by_tier = conn.execute(
                f"SELECT tier, COUNT(*), COALESCE(SUM(estimated_cost_usd), 0) "
                f"FROM cost_events {where} GROUP BY tier",
                params,
            ).fetchall()

            return {
                "total_requests": total_requests,
                "total_cost_usd": total_cost,
                "cache_hits": cache_hits,
                "cache_hit_rate": (cache_hits / total_requests) if total_requests else 0.0,
                "by_tier": {tier: {"requests": count, "cost_usd": cost} for tier, count, cost in by_tier},
            }
        finally:
            conn.close()

    async def summary(self, api_key: str | None = None, since_ts: float | None = None) -> dict:
        return await asyncio.to_thread(self._summary_sync, api_key, since_ts)

    def _timeseries_sync(self, since_ts: float, bucket_seconds: int) -> list[dict]:
        conn = sqlite3.connect(self._db_path)
        try:
            # Bucket by floor(ts / bucket_seconds) rather than a SQL date
            # function -- keeps this independent of bucket_seconds being a
            # "nice" unit like an hour or a day, and avoids relying on
            # sqlite's unixepoch datetime formatting.
            rows = conn.execute(
                "SELECT CAST(ts / ? AS INTEGER) AS bucket, COUNT(*), "
                "COALESCE(SUM(estimated_cost_usd), 0), COALESCE(SUM(cache_hit), 0) "
                "FROM cost_events WHERE ts >= ? GROUP BY bucket ORDER BY bucket",
                (bucket_seconds, since_ts),
            ).fetchall()
            return [
                {
                    "bucket_start_ts": bucket * bucket_seconds,
                    "requests": count,
                    "cost_usd": cost,
                    "cache_hits": cache_hits,
                }
                for bucket, count, cost, cache_hits in rows
            ]
        finally:
            conn.close()

    async def timeseries(self, since_ts: float, bucket_seconds: int = 3600) -> list[dict]:
        return await asyncio.to_thread(self._timeseries_sync, since_ts, bucket_seconds)

    def _recent_events_sync(self, limit: int) -> list[dict]:
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT ts, api_key, tier, provider, model, estimated_cost_usd, cache_hit, latency_ms "
                "FROM cost_events ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "ts": ts,
                    "api_key": api_key,
                    "tier": tier,
                    "provider": provider,
                    "model": model,
                    "estimated_cost_usd": cost,
                    "cache_hit": bool(cache_hit),
                    "latency_ms": latency_ms,
                }
                for ts, api_key, tier, provider, model, cost, cache_hit, latency_ms in rows
            ]
        finally:
            conn.close()

    async def recent_events(self, limit: int = 50) -> list[dict]:
        return await asyncio.to_thread(self._recent_events_sync, limit)


_ledger_singleton: CostLedger | None = None


def get_cost_ledger() -> CostLedger:
    global _ledger_singleton
    if _ledger_singleton is None:
        _ledger_singleton = CostLedger(get_settings().cost_ledger_path)
    return _ledger_singleton
