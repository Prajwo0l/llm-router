"""
Wraps the fine-tuned difficulty classifier -- with a hard requirement: this
module must be safely importable and usable even when no checkpoint exists
and even when torch/transformers aren't installed at all. That's what lets
the rest of the router (cache, providers, rate limiting, benchmark harness)
be built and reasoned about locally without pulling in any ML runtime.

torch/transformers are imported lazily, inside `_load_model`, only once a
checkpoint directory is confirmed to exist on disk -- so `import
app.router.classifier` never fails just because those packages aren't
installed, and the API server boots fine in pure-heuristic mode.
"""
from __future__ import annotations

import json
import logging
import os
import threading

from app.config import get_settings
from app.router.heuristic import DifficultyResult, score_heuristic

logger = logging.getLogger(__name__)

_LABELS = ["easy", "medium", "hard"]


class DifficultyClassifier:
    """Singleton-ish wrapper: attempts to load a fine-tuned checkpoint once,
    lazily, on first use. If the checkpoint directory doesn't exist, or
    loading fails for any reason (missing torch, corrupted checkpoint,
    wrong label count), it logs why and every subsequent call transparently
    falls back to the heuristic scorer -- callers never need to know which
    path served a given request; DifficultyResult.source tells them.
    """

    def __init__(self, checkpoint_dir: str | None = None):
        self._checkpoint_dir = checkpoint_dir or get_settings().classifier_checkpoint_dir
        self._model = None
        self._tokenizer = None
        self._label_map: list[str] = _LABELS
        self._load_attempted = False
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        """Runs at most once. Any failure here is logged and swallowed --
        this must never raise up into a request path; falling back to the
        heuristic is always a valid outcome, a crashed server is not."""
        with self._load_lock:
            if self._load_attempted:
                return
            self._load_attempted = True

            if not self._checkpoint_dir or not os.path.isdir(self._checkpoint_dir):
                logger.info(
                    "No difficulty-classifier checkpoint found at %s -- using the heuristic scorer. "
                    "Train one in Colab (see training/) and point classifier_checkpoint_dir at it "
                    "to switch this over.",
                    self._checkpoint_dir,
                )
                return

            try:
                import torch  # noqa: F401  -- imported only to fail fast with a clear message if missing
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                logger.warning(
                    "Checkpoint directory %s exists but torch/transformers aren't installed (%s) -- "
                    "falling back to the heuristic scorer. Install the [ml] extra to use the trained classifier.",
                    self._checkpoint_dir,
                    exc,
                )
                return

            try:
                label_map_path = os.path.join(self._checkpoint_dir, "label_map.json")
                if os.path.exists(label_map_path):
                    with open(label_map_path) as f:
                        # Expected shape: {"0": "easy", "1": "medium", "2": "hard"}
                        # written by training/train_classifier.py -- see that file for why
                        # the label order must never be assumed, only read.
                        raw = json.load(f)
                        self._label_map = [raw[str(i)] for i in range(len(raw))]

                self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint_dir)
                self._model = AutoModelForSequenceClassification.from_pretrained(self._checkpoint_dir)
                self._model.eval()
                logger.info("Loaded difficulty classifier from %s (labels: %s)", self._checkpoint_dir, self._label_map)
            except Exception:
                logger.exception(
                    "Failed to load difficulty classifier from %s -- falling back to the heuristic scorer.",
                    self._checkpoint_dir,
                )
                self._model = None
                self._tokenizer = None

    def score(self, prompt: str) -> DifficultyResult:
        if not self._load_attempted:
            self._load_model()

        if self._model is None or self._tokenizer is None:
            return score_heuristic(prompt)

        import torch  # safe: only reached once _load_model has already succeeded

        with torch.no_grad():
            inputs = self._tokenizer(prompt, truncation=True, max_length=256, return_tensors="pt")
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            top_idx = int(torch.argmax(probs).item())
            confidence = float(probs[top_idx].item())

        label = self._label_map[top_idx] if top_idx < len(self._label_map) else "medium"
        return DifficultyResult(label=label, confidence=confidence, source="classifier")


_classifier_singleton: DifficultyClassifier | None = None


def get_classifier() -> DifficultyClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = DifficultyClassifier()
    return _classifier_singleton
