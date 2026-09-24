"""
Semantic cache: embed each prompt, look up the nearest previously-seen
prompt in a FAISS index, and reuse its stored response if the cosine
similarity clears `semantic_cache_similarity_threshold`. This is the
single biggest lever on cost for a workload with repeated or near-duplicate
questions (FAQ-style traffic, retries, near-identical support tickets) --
and the threshold is exactly the kind of thing that should be tuned against
the benchmark's cache-hit-rate / quality tradeoff, not guessed at.

faiss and the embedding backends are imported lazily so this module (and
the FastAPI app that imports it) can be loaded without those packages
installed -- the cache is simply disabled (see get_cache()) rather than
crashing the server, matching the same "heavy deps are optional, not
required to boot" pattern used in router/classifier.py.
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    prompt: str
    response: str
    tier_served: str
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


@dataclass
class ClosestMatch:
    """The nearest stored entry to a query, whether or not it clears the
    similarity threshold. Always returning the closest match (not just on
    a hit) is what makes the threshold tunable by *observation* -- a miss
    that scored 0.91 against a 0.94 threshold tells you something a bare
    "no hit" doesn't; see RouterMetadata.closest_cache_similarity, which
    surfaces this value on every real (non-cached) response."""

    entry: CacheEntry
    similarity: float

    def is_hit(self, threshold: float) -> bool:
        return self.similarity >= threshold


class SemanticCache:
    def __init__(self, dim: int, max_entries: int, similarity_threshold: float):
        import faiss  # local import -- see module docstring

        self._faiss = faiss
        self._dim = dim
        self._max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        # IndexFlatIP over L2-normalized vectors == cosine similarity search.
        # Flat (brute-force) is intentional: exact, simple to reason about,
        # and fast enough up to tens of thousands of entries -- swap for an
        # IVF/HNSW index only if the benchmark shows lookup latency actually
        # matters at your real cache size, not preemptively.
        self._index = faiss.IndexFlatIP(dim)
        self._entries: list[CacheEntry] = []

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def lookup(self, embedding: np.ndarray) -> ClosestMatch | None:
        """Returns the closest stored entry regardless of whether it clears
        `similarity_threshold` -- None only when the cache is empty. Callers
        that only care about actual hits should check `.is_hit(threshold)`;
        this module intentionally doesn't decide hit/miss itself, so a
        near-miss's similarity is never thrown away before the caller can
        see it."""
        if self._index.ntotal == 0:
            return None
        query = self._normalize(embedding).astype("float32").reshape(1, -1)
        similarities, indices = self._index.search(query, k=1)
        best_sim = float(similarities[0][0])
        best_idx = int(indices[0][0])
        if best_idx < 0:
            return None
        return ClosestMatch(entry=self._entries[best_idx], similarity=best_sim)

    def store(self, embedding: np.ndarray, prompt: str, response: str, tier_served: str) -> None:
        if len(self._entries) >= self._max_entries:
            logger.warning("Semantic cache at max_entries=%d -- not storing new entry. "
                            "Consider an eviction policy before relying on this at scale.", self._max_entries)
            return
        vec = self._normalize(embedding).astype("float32").reshape(1, -1)
        self._index.add(vec)
        self._entries.append(CacheEntry(prompt=prompt, response=response, tier_served=tier_served))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._faiss.write_index(self._index, f"{path}.faiss")
        with open(f"{path}.entries.pkl", "wb") as f:
            pickle.dump(self._entries, f)

    def load(self, path: str) -> bool:
        if not os.path.exists(f"{path}.faiss"):
            return False
        self._index = self._faiss.read_index(f"{path}.faiss")
        with open(f"{path}.entries.pkl", "rb") as f:
            self._entries = pickle.load(f)
        return True

    @property
    def size(self) -> int:
        return len(self._entries)


class Embedder:
    """Pluggable embedding backend. Both branches import their heavy
    dependency lazily, on first use, not at class-construction time."""

    def __init__(self, backend: str, model: str):
        self._backend = backend
        self._model_name = model
        self._openai_client = None
        self._st_model = None

    @property
    def dim(self) -> int:
        # Known dims for the models this project actually uses -- if you
        # swap embedding_model in config.py, update this (or compute it
        # once via a real embed() call at startup instead of hardcoding).
        known_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "all-MiniLM-L6-v2": 384,
        }
        if self._model_name not in known_dims:
            raise ValueError(
                f"Unknown embedding dimension for model '{self._model_name}' -- add it to Embedder.dim's "
                f"known_dims table (or compute it dynamically) before using this model."
            )
        return known_dims[self._model_name]

    async def embed(self, text: str) -> np.ndarray:
        if self._backend == "openai":
            return await self._embed_openai(text)
        elif self._backend == "sentence-transformers":
            return self._embed_sentence_transformers(text)
        raise ValueError(f"Unknown embedding backend: {self._backend}")

    async def _embed_openai(self, text: str) -> np.ndarray:
        if self._openai_client is None:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(api_key=get_settings().openai_api_key)
        response = await self._openai_client.embeddings.create(model=self._model_name, input=text)
        return np.array(response.data[0].embedding, dtype="float32")

    def _embed_sentence_transformers(self, text: str) -> np.ndarray:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self._model_name)
        return self._st_model.encode(text, convert_to_numpy=True).astype("float32")


_cache_singleton: SemanticCache | None = None
_embedder_singleton: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder_singleton
    if _embedder_singleton is None:
        settings = get_settings()
        _embedder_singleton = Embedder(backend=settings.embedding_backend, model=settings.embedding_model)
    return _embedder_singleton


def get_cache() -> SemanticCache | None:
    """Returns None (cache disabled) rather than raising if faiss isn't
    installed or semantic_cache_enabled is False -- callers must handle
    the None case, which keeps "cache is optional" enforced at the type
    level instead of by convention."""
    global _cache_singleton
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        return None
    if _cache_singleton is None:
        try:
            embedder = get_embedder()
            _cache_singleton = SemanticCache(
                dim=embedder.dim,
                max_entries=settings.semantic_cache_max_entries,
                similarity_threshold=settings.semantic_cache_similarity_threshold,
            )
        except ImportError:
            logger.warning("faiss is not installed -- semantic cache disabled. Install the [cache] extra to enable it.")
            return None
    return _cache_singleton
