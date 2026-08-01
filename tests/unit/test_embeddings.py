"""Unit tests for the BGE embedding wrapper."""

from __future__ import annotations

import importlib
import sys
import types
from typing import List

import pytest


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Install fake `sentence_transformers` and `torch` modules in sys.modules."""
    encoded_calls: List[List[str]] = []

    class _FakeNumpyVector:
        def __init__(self, data):
            self._data = list(data)

        def tolist(self):
            return list(self._data)

    class _FakeTransformer:
        def __init__(self, model_name, device):
            self.model_name = model_name
            self.device = device

        def encode(self, texts, convert_to_numpy=False, normalize_embeddings=False, show_progress_bar=False):
            encoded_calls.append(list(texts))
            return [_FakeNumpyVector([float(i), 0.5]) for i in range(len(texts))]

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _FakeTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import src.memory.embeddings as embeddings_module

    embeddings_module = importlib.reload(embeddings_module)
    return embeddings_module, encoded_calls


def test_warmup_loads_on_cuda_when_available(fake_sentence_transformers):
    embeddings_module, _calls = fake_sentence_transformers
    assert embeddings_module.is_ready() is False

    embeddings_module.warmup_embedding_function()

    assert embeddings_module.is_ready() is True
    assert embeddings_module.is_loading() is False
    assert embeddings_module.last_error() is None
    ef = embeddings_module.get_embedding_function()
    assert ef.device == "cuda"
    assert ef.name() == "bge:BAAI/bge-large-en-v1.5:cuda"


def test_warmup_falls_back_to_cpu_when_cuda_unavailable(fake_sentence_transformers, monkeypatch):
    embeddings_module, _calls = fake_sentence_transformers
    sys.modules["torch"].cuda.is_available = lambda: False

    embeddings_module.warmup_embedding_function()

    ef = embeddings_module.get_embedding_function()
    assert ef.device == "cpu"


def test_warmup_is_idempotent(fake_sentence_transformers):
    embeddings_module, _calls = fake_sentence_transformers
    embeddings_module.warmup_embedding_function()
    first = embeddings_module.get_embedding_function()
    embeddings_module.warmup_embedding_function()
    assert embeddings_module.get_embedding_function() is first


def test_get_embedding_function_raises_before_warmup(fake_sentence_transformers):
    embeddings_module, _calls = fake_sentence_transformers
    with pytest.raises(RuntimeError, match="not loaded"):
        embeddings_module.get_embedding_function()


def test_call_embedding_function_returns_vectors(fake_sentence_transformers):
    embeddings_module, calls = fake_sentence_transformers
    embeddings_module.warmup_embedding_function()
    ef = embeddings_module.get_embedding_function()

    vectors = ef(["hello", "world"])
    assert vectors == [[0.0, 0.5], [1.0, 0.5]]
    assert calls[-1] == ["hello", "world"]


def test_call_embedding_function_coerces_non_list_input(fake_sentence_transformers):
    embeddings_module, calls = fake_sentence_transformers
    embeddings_module.warmup_embedding_function()
    ef = embeddings_module.get_embedding_function()

    vectors = ef("just-a-string")
    assert len(vectors) == 1
    assert calls[-1] == ["just-a-string"]


def test_warmup_records_error_on_failure(monkeypatch):
    fake_st = types.ModuleType("sentence_transformers")

    def _fail(*_a, **_kw):
        raise RuntimeError("boom")

    fake_st.SentenceTransformer = _fail
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import src.memory.embeddings as embeddings_module

    embeddings_module = importlib.reload(embeddings_module)

    with pytest.raises(RuntimeError, match="boom"):
        embeddings_module.warmup_embedding_function()

    assert embeddings_module.is_ready() is False
    assert isinstance(embeddings_module.last_error(), RuntimeError)
