"""
Minimal metrics dashboard: one HTML page (static, vanilla JS, Chart.js from
a CDN -- no frontend build step, consistent with this project's "no heavy
job" philosophy) plus a small JSON API it polls.

This is an aggregate, all-keys view (unlike GET /v1/usage in app/main.py,
which is scoped to the caller's own api_key). There's no auth system in
this project (see app/main.py's _resolve_api_key docstring), so this
endpoint is unauthenticated -- fine for local/single-operator use, but
put it behind your own auth or a reverse-proxy basic-auth rule before
exposing this port publicly. That's a deliberate scope cut, not an
oversight: building a real auth system was out of scope for the same
reason streaming was (see docs/architecture.md).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.middleware.cost_tracking import get_cost_ledger

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_STATIC_DIR = Path(__file__).parent / "static"


@router.get("")
async def dashboard_page():
    return FileResponse(_STATIC_DIR / "index.html")


@router.get("/api/summary")
async def dashboard_summary(window_hours: int = 24, bucket_minutes: int = 60):
    """Everything the dashboard page needs in one round trip: totals,
    per-tier breakdown, a cost/requests time series, and the most recent
    events -- one call rather than the page firing off several."""
    ledger = get_cost_ledger()
    since_ts = time.time() - window_hours * 3600

    summary, timeseries, recent = await asyncio.gather(
        ledger.summary(since_ts=since_ts),
        ledger.timeseries(since_ts=since_ts, bucket_seconds=bucket_minutes * 60),
        ledger.recent_events(limit=25),
    )

    return {
        "window_hours": window_hours,
        "generated_at": time.time(),
        "summary": summary,
        "timeseries": timeseries,
        "recent_events": recent,
    }
