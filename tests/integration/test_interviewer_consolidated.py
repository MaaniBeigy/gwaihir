"""Live LLM tests for the consolidated-question interviewer.

Skipped unless `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) is set; selected with
`-m live` since they call the real LLM endpoint and cost tokens.
"""

from __future__ import annotations

import os
import re

import pytest

from src.agents.interviewer_core import InterviewerAgent

# These chain many real LLM calls, so they get a longer per-test timeout than the offline default.
pytestmark = [pytest.mark.live, pytest.mark.timeout(300)]


# A unique counter so each test gets its own in-memory Chroma collection.
_collection_counter = {"n": 0}


def _has_live_keys() -> bool:
    has_openai = bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("OPENAI_MODEL"))
    openrouter = os.getenv("OPENROUTER_API_KEY", "")
    has_openrouter = bool(openrouter) and "dummy" not in openrouter.lower()
    return has_openai or has_openrouter


pytest_plugins: list = []


@pytest.fixture(autouse=True)
def _skip_when_no_keys():
    if not _has_live_keys():
        pytest.skip("OPENAI_API_KEY+OPENAI_MODEL (or OPENROUTER_API_KEY) not set; skipping live LLM tests")


class _InMemoryCollection:
    """No-op Chroma collection stub; the interviewer drives history in memory."""

    def add(self, *args, **kwargs):
        return None

    def query(self, *args, **kwargs):
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

    def get(self, *args, **kwargs):
        return {"documents": [], "metadatas": [], "ids": []}


def _fresh_agent() -> InterviewerAgent:
    _collection_counter["n"] += 1
    return InterviewerAgent(collection=_InMemoryCollection())


def _send(agent: InterviewerAgent, user_msg: str) -> str:
    """Send a user message and return the assistant reply."""
    agent.conversation_history.append({"role": "user", "content": user_msg})
    reply = agent._send_to_llm(agent.conversation_history)
    agent.conversation_history.append({"role": "assistant", "content": reply})
    return reply


def _bootstrap_to_q9_for_gym(agent: InterviewerAgent) -> str:
    """Drive Q1 to Q8 with terse answers; return the Q9 question for gym."""
    reply = _send(agent, "Start interview about habits and routines.")
    # Section 1: sleep and eat.
    reply = _send(agent, "07:00")
    reply = _send(agent, "07:30, 15 min")
    reply = _send(agent, "12:00, 30 min")
    reply = _send(agent, "18:00, 30 min")
    reply = _send(agent, "23:00")
    # Section 2: work.
    reply = _send(agent, "08:00")
    reply = _send(agent, "17:00")
    # Section 3: sports.
    reply = _send(agent, "gym, walking")
    return reply


def test_q9_is_a_single_consolidated_question() -> None:
    """Q9 must ask day, time, and duration in one sentence with the sport name."""
    agent = _fresh_agent()
    q9 = _bootstrap_to_q9_for_gym(agent)

    lower = q9.lower()
    assert "gym" in lower, f"Q9 should mention 'gym' explicitly; got: {q9!r}"
    # Must touch days AND time AND duration.
    assert ("day" in lower) and (
        "time" in lower or "when" in lower
    ), f"Q9 must ask about days and time; got: {q9!r}"
    assert (
        "how long" in lower or "duration" in lower or "for how long" in lower
    ), f"Q9 must ask about duration; got: {q9!r}"


def test_multi_row_reply_is_accepted_and_interviewer_advances() -> None:
    """After a multi-row Q9 reply for gym, the interviewer must advance, not re-ask gym."""
    agent = _fresh_agent()
    _ = _bootstrap_to_q9_for_gym(agent)
    next_q = _send(agent, "Tue 19:30 2h, Thu 19:30 2h, Sat 16:30 2h")
    lower = next_q.lower()
    # Must not still be asking about gym
    assert "gym" not in lower, (
        f"Interviewer kept asking about gym after a complete reply; " f"got: {next_q!r}"
    )
    # And the reply must be a new question (single sentence ending with '?'
    # or starting with 'Q'); not just an acknowledgement.
    assert "?" in next_q, f"Expected a follow-up question; got: {next_q!r}"


def test_flexible_time_phrase_with_duration_is_accepted() -> None:
    """A flexible time phrase plus a duration is a complete answer; no HH:MM required."""
    agent = _fresh_agent()
    _ = _bootstrap_to_q9_for_gym(agent)
    next_q = _send(agent, "evenings after dinner for 1.5 hours, Mon and Wed")
    lower = next_q.lower()
    # The interviewer must move on, not press for an exact HH:MM on the same sport.
    assert "gym" not in lower, (
        f"Interviewer pressed for exact time on gym after a flex answer; " f"got: {next_q!r}"
    )


def test_unparseable_reply_triggers_one_clarification_only() -> None:
    """An unparseable reply may trigger the Q10 clarification once; the next valid reply advances."""
    agent = _fresh_agent()
    _ = _bootstrap_to_q9_for_gym(agent)
    clar = _send(agent, "yeah sometimes")
    follow = _send(agent, "Tuesdays in the morning for an hour")
    lower = follow.lower()
    assert "gym" not in lower, (
        f"Interviewer kept asking about gym after a valid clarification reply; "
        f"got clarification={clar!r}; follow={follow!r}"
    )


def test_per_sport_then_per_hobby_round_trips() -> None:
    """Two sports with multi-row replies, then move into hobbies within a bounded number of turns."""
    agent = _fresh_agent()
    _ = _bootstrap_to_q9_for_gym(agent)
    second = _send(agent, "Tue 19:30 2h, Thu 19:30 2h, Sat 16:30 2h")
    assert "walk" in second.lower(), f"Expected next Q9 about 'walking'; got: {second!r}"
    third = _send(agent, "Mon, Wed, Fri at 18:30 for 1 hour")
    text = third.lower()
    assert any(
        word in text for word in ("hobb", "hobby", "reading", "gaming")
    ), f"Expected Q11 (hobbies) after the last sport; got: {third!r}"
