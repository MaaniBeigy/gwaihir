"""Defensive paths in `InterviewerAgent`: instruction loading, fallback returns, history shape mismatches."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.agents.interviewer_core import InterviewerAgent


class _StubCollection:
    def add(self, documents, ids, metadatas):
        pass


@pytest.fixture(autouse=True)
def _stable_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setattr("src.agents.interviewer_core.time.sleep", lambda _s: None)
    monkeypatch.setattr("src.agents.interviewer_core.store_qa", lambda *a, **kw: None)


# --------------------------------------------------------------------------------------------
# _load_instructions: missing file
# --------------------------------------------------------------------------------------------


def test_load_instructions_raises_when_template_file_missing(monkeypatch):
    """An absent `Interviewer_instructions.txt` is reported as a clear startup error."""
    monkeypatch.setattr("src.agents.interviewer_core.os.path.exists", lambda path: False)
    with pytest.raises(RuntimeError, match="Interviewer instructions not found"):
        InterviewerAgent(collection=_StubCollection())


# --------------------------------------------------------------------------------------------
# _send_to_llm: returns the fallback after three empty responses (covers the False arm of
# `if attempt < EMPTY_RESPONSE_RETRY_ATTEMPTS` and the trailing `return ... FALLBACK` line)
# --------------------------------------------------------------------------------------------


class _MockResponse:
    def __init__(self, content: str, status_code: int = 200):
        self._content = content
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_chat_returns_fallback_after_three_empty_responses(monkeypatch):
    """All three inner-loop attempts return whitespace; the agent surfaces the
    canned fallback message instead of looping forever."""
    calls = []

    def _post(*args, **kwargs):
        calls.append(1)
        return _MockResponse("   ")

    monkeypatch.setattr("requests.post", _post)
    agent = InterviewerAgent(collection=_StubCollection())
    reply = agent.chat("hi", clear_history=True)
    assert reply == InterviewerAgent.EMPTY_RESPONSE_FALLBACK
    # Three attempts on the first (and only) HTTP cycle.
    assert len(calls) == 3


# --------------------------------------------------------------------------------------------
# chat: skips the QA-store block when no collection is attached
# --------------------------------------------------------------------------------------------


def test_chat_skips_qa_storage_when_collection_is_none(monkeypatch):
    """An agent constructed without a collection does not attempt to call
    `store_qa` and still returns the LLM reply."""
    monkeypatch.setattr("requests.post", lambda *a, **kw: _MockResponse("Q1: hi"))

    store_calls = []
    monkeypatch.setattr(
        "src.agents.interviewer_core.store_qa",
        lambda *a, **kw: store_calls.append(1),
    )

    agent = InterviewerAgent(collection=None, llm_api_key="sk-or-v1-test")
    reply = agent.chat("hello", clear_history=True)
    assert reply == "Q1: hi"
    assert store_calls == []


# get_history with a non-list documents field returns an empty result.


def test_get_history_returns_empty_when_documents_field_is_not_a_list(monkeypatch):
    """When the ChromaDB result has a `documents` value that isn't a list,
    history retrieval returns an empty list rather than crashing."""
    monkeypatch.setattr(
        "src.agents.interviewer_core.retrieve_all_history",
        lambda col: {"documents": "this is a string, not a list", "metadatas": []},
    )
    agent = InterviewerAgent(collection=_StubCollection())
    assert agent.get_history() == []


# get_history with more documents than metadatas hits the idx-out-of-range branch.


def test_get_history_handles_documents_without_matching_metadatas(monkeypatch):
    """When metadatas is shorter than documents, missing indices skip the
    timestamp parse and fall straight through to the entry append."""
    monkeypatch.setattr(
        "src.agents.interviewer_core.retrieve_all_history",
        lambda col: {
            "documents": ["first entry", "second entry", "third entry"],
            # Only one metadata; indices 1 and 2 trigger the
            # `idx < len(metadatas)` False arm.
            "metadatas": [{"timestamp": "2026-04-15T10:00:00Z"}],
        },
    )
    agent = InterviewerAgent(collection=_StubCollection())
    history = agent.get_history()
    assert "first entry" in history
    assert "second entry" in history
    assert "third entry" in history


def test_get_history_handles_metadata_entry_that_is_not_a_dict(monkeypatch):
    """A metadata entry that's not a dict at the matching index also takes the
    skip-timestamp branch."""
    monkeypatch.setattr(
        "src.agents.interviewer_core.retrieve_all_history",
        lambda col: {
            "documents": ["only entry"],
            "metadatas": ["not-a-dict"],
        },
    )
    agent = InterviewerAgent(collection=_StubCollection())
    assert agent.get_history() == ["only entry"]
