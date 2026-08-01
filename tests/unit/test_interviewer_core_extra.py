"""`InterviewerAgent` 429 retry, HTTP error propagation, and helper classifiers."""

from __future__ import annotations

import pytest
import requests

from src.agents.interviewer_core import InterviewerAgent


class _StubCollection:
    def add(self, documents, ids, metadatas):
        pass


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setattr("src.agents.interviewer_core.time.sleep", lambda _s: None)
    monkeypatch.setattr("src.agents.interviewer_core.store_qa", lambda *a, **kw: None)


class _MockResponse:
    def __init__(self, content, status_code=200):
        self._content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_chat_retries_on_429_then_succeeds(monkeypatch):
    calls = []

    def _post(url, *args, **kwargs):
        calls.append(url)
        if len(calls) < 3:
            return _MockResponse("", status_code=429)
        return _MockResponse("Q1: when do you wake up?")

    monkeypatch.setattr("requests.post", _post)
    agent = InterviewerAgent(collection=_StubCollection())
    reply = agent.chat("hi", clear_history=True)
    assert reply == "Q1: when do you wake up?"
    assert len(calls) == 3


def test_chat_raises_on_non_429_http_error(monkeypatch):
    def _post(*args, **kwargs):
        return _MockResponse("", status_code=500)

    monkeypatch.setattr("requests.post", _post)
    agent = InterviewerAgent(collection=_StubCollection())
    with pytest.raises(requests.exceptions.HTTPError):
        agent.chat("hi", clear_history=True)


def test_chat_returns_fallback_after_max_429_retries(monkeypatch):
    def _post(*args, **kwargs):
        return _MockResponse("", status_code=429)

    monkeypatch.setattr("requests.post", _post)
    agent = InterviewerAgent(collection=_StubCollection())
    reply = agent.chat("hi", clear_history=True)
    assert "didn't get a usable response" in reply.lower()


def test_extract_question_number_recognises_q_prefix():
    assert InterviewerAgent._extract_question_number("Q5: tell me more") == "Q5"
    assert InterviewerAgent._extract_question_number("**3.** habit?") == "Q3"
    assert InterviewerAgent._extract_question_number("no number here") is None


def test_extract_section_classifies_keywords():
    assert InterviewerAgent._extract_section("Section 1 sleep") == "sleep_eat_routines"
    assert InterviewerAgent._extract_section("section 2 work and study") == "work_study"
    assert InterviewerAgent._extract_section("hobbies and sports") == "hobbies_sports"
    assert InterviewerAgent._extract_section("commute") == "commute"
    assert InterviewerAgent._extract_section("weekend plans") == "weekend"
    assert InterviewerAgent._extract_section("nothing matches") is None


def test_reset_clears_conversation_history(monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **kw: _MockResponse("Q1: hi"),
    )
    agent = InterviewerAgent(collection=_StubCollection())
    agent.chat("hello", clear_history=True)
    assert agent.get_conversation_history()
    agent.reset()
    assert agent.get_conversation_history() == []


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        InterviewerAgent(collection=_StubCollection(), llm_api_key=None)


def test_get_history_returns_empty_when_no_collection():
    agent = InterviewerAgent(collection=None, llm_api_key="sk-or-v1-test")
    assert agent.get_history() == []


def test_get_history_swallows_exceptions(monkeypatch):
    monkeypatch.setattr(
        "src.agents.interviewer_core.retrieve_all_history",
        lambda col: (_ for _ in ()).throw(RuntimeError("read failed")),
    )
    agent = InterviewerAgent(collection=_StubCollection(), llm_api_key="sk-or-v1-test")
    assert agent.get_history() == []
