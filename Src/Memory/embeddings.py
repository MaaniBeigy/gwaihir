"""GPU embedding function for the coach Chroma collections.

Loads `BAAI/bge-large-en-v1.5` on CUDA when available, falling back to CPU otherwise.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

logger = logging.getLogger("coach.embeddings")

_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
_state_lock = threading.Lock()
_loading = False
_ready = False
_load_error: Optional[Exception] = None
_embedding_function = None


def is_ready() -> bool:
    return _ready


def is_loading() -> bool:
    return _loading


def last_error() -> Optional[Exception]:
    return _load_error


class _BGEEmbeddingFunction:
    """Chroma-compatible embedding function wrapping a SentenceTransformer model."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            logger.warning(
                "CUDA not available; loading embedding model %s on CPU.",
                model_name,
            )
            self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not isinstance(input, list):
            input = [str(input)]
        # normalize_embeddings keeps cosine similarity sensible for BGE.
        vectors = self.model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]

    def name(self) -> str:
        return f"bge:{self.model_name}:{self.device}"


def get_embedding_function():
    """Return the singleton embedding function or raise when not loaded yet."""
    if _embedding_function is None:
        raise RuntimeError("Embedding model is not loaded yet. /health returns 503 until startup completes.")
    return _embedding_function


def warmup_embedding_function(model_name: str = _DEFAULT_MODEL) -> None:
    """Load the embedding model. Safe to call multiple times; only the first call loads weights."""
    global _loading, _ready, _load_error, _embedding_function

    with _state_lock:
        if _ready or _loading:
            return
        _loading = True
        _load_error = None

    try:
        logger.info("Loading embedding model %s", model_name)
        ef = _BGEEmbeddingFunction(model_name=model_name)
        with _state_lock:
            _embedding_function = ef
            _ready = True
        logger.info("Embedding model %s loaded on device=%s", model_name, ef.device)
    except Exception as exc:  # noqa: BLE001
        with _state_lock:
            _load_error = exc
        logger.exception("Failed to load embedding model %s: %s", model_name, exc)
        raise
    finally:
        with _state_lock:
            _loading = False
