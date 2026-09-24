"""
Tracing/metrics, off by default (otel_enabled / langsmith_enabled both
default False in config.py). Both SDKs are imported lazily so the app
runs fine with neither installed -- tracing is an add-on, not a boot
requirement, same pattern as the classifier and semantic cache.

This module exposes one function, `trace_completion`, used as an async
context manager around a single routed completion. It's deliberately thin:
it does not try to be a generic tracing abstraction, just enough to get
router decisions (tier, cache hit, cost, latency) into a trace span so
they're visible in whichever backend is enabled.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from app.config import get_settings

logger = logging.getLogger(__name__)

_otel_tracer = None
_langsmith_client = None


def _get_otel_tracer():
    global _otel_tracer
    if _otel_tracer is None:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        settings = get_settings()
        provider = TracerProvider(resource=Resource.create({"service.name": "llm-router"}))
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _otel_tracer = trace.get_tracer("llm-router")
    return _otel_tracer


def _get_langsmith_client():
    global _langsmith_client
    if _langsmith_client is None:
        from langsmith import Client

        _langsmith_client = Client(api_key=get_settings().langsmith_api_key)
    return _langsmith_client


@asynccontextmanager
async def trace_completion(name: str, metadata: dict):
    """Usage:
        async with trace_completion("chat_completion", {"tier": "cheap"}) as span:
            ... do the work ...
            span.set("cache_hit", True)

    `span` is a tiny wrapper object (see _SpanHandle below), not the raw
    SDK span, so callers don't need to know which backend (or neither) is
    active.
    """
    settings = get_settings()
    started_at = time.time()
    handle = _SpanHandle(extra=dict(metadata))

    otel_span_cm = None
    if settings.otel_enabled:
        try:
            tracer = _get_otel_tracer()
            otel_span_cm = tracer.start_as_current_span(name)
            otel_span_cm.__enter__()
        except Exception:
            logger.exception("Failed to start OpenTelemetry span -- continuing without tracing")
            otel_span_cm = None

    try:
        yield handle
    finally:
        duration_ms = (time.time() - started_at) * 1000
        handle.extra["duration_ms"] = duration_ms

        if otel_span_cm is not None:
            try:
                from opentelemetry import trace as otel_trace

                current_span = otel_trace.get_current_span()
                for key, value in handle.extra.items():
                    current_span.set_attribute(key, value)
                otel_span_cm.__exit__(None, None, None)
            except Exception:
                logger.exception("Failed to finalize OpenTelemetry span")

        if settings.langsmith_enabled:
            try:
                client = _get_langsmith_client()
                client.create_run(
                    name=name,
                    run_type="chain",
                    inputs={k: v for k, v in handle.extra.items() if k != "duration_ms"},
                    outputs={"duration_ms": duration_ms},
                    project_name=settings.langsmith_project,
                )
            except Exception:
                logger.exception("Failed to log run to LangSmith -- continuing without it")


class _SpanHandle:
    def __init__(self, extra: dict):
        self.extra = extra

    def set(self, key: str, value) -> None:
        self.extra[key] = value
