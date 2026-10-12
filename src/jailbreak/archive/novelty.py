"""Embedding-based novelty filter (drop near-duplicates before executor call)."""

from __future__ import annotations

import struct

import numpy as np


class NoveltyFilter:
    """Cosine-similarity dedup against per-objective archive of embeddings.

    Uses fastembed if available. Falls back to a cheap hashing embedding for
    offline tests (still gives real cosine similarities on token overlap).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", threshold: float = 0.92) -> None:
        self.threshold = threshold
        self._backend: _Backend
        try:
            from fastembed import TextEmbedding  # type: ignore[import-untyped]

            self._backend = _FastembedBackend(TextEmbedding(model_name=model_name))
        except Exception:
            self._backend = _HashingBackend()

    def encode(self, text: str) -> np.ndarray:
        return self._backend.encode(text)

    @staticmethod
    def to_bytes(vec: np.ndarray) -> bytes:
        arr = vec.astype(np.float32)
        return struct.pack(f"<{len(arr)}f", *arr)

    @staticmethod
    def from_bytes(data: bytes) -> np.ndarray:
        n = len(data) // 4
        return np.array(struct.unpack(f"<{n}f", data), dtype=np.float32)

    def is_duplicate(self, candidate: np.ndarray, known: list[np.ndarray]) -> bool:
        if not known:
            return False
        c = candidate / (np.linalg.norm(candidate) + 1e-9)
        for k in known:
            kn = k / (np.linalg.norm(k) + 1e-9)
            sim = float(np.dot(c, kn))
            if sim >= self.threshold:
                return True
        return False


class _Backend:
    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError


class _FastembedBackend(_Backend):
    def __init__(self, model) -> None:  # type: ignore[no-untyped-def]
        self.model = model

    def encode(self, text: str) -> np.ndarray:
        vec = next(iter(self.model.embed([text])))
        return np.asarray(vec, dtype=np.float32)


class _HashingBackend(_Backend):
    """Cheap fallback: bag-of-token-hashes → 384-dim float vector."""

    DIM = 384

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.DIM, dtype=np.float32)
        for tok in text.lower().split():
            h = hash(tok) % self.DIM
            vec[h] += 1.0
        return vec
