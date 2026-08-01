"""Live LLM checks for the coach flow.

Skipped unless `OPENAI_API_KEY`+`OPENAI_MODEL` or a real `OPENROUTER_API_KEY` is set.
These chain several real LLM calls, so they get a longer per-test timeout.
"""

from __future__ import annotations

import os

import pytest

from src.agents import coach_graph


def _live_key_available() -> bool:
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"):
        return True
    key = os.getenv("OPENROUTER_API_KEY", "")
    return bool(key) and "dummy" not in key.lower()


RUN_LIVE = _live_key_available()

pytestmark = [
    pytest.mark.live,
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="set OPENAI_API_KEY+OPENAI_MODEL or a real OPENROUTER_API_KEY to run live LLM tests",
    ),
]


def _cfg():
    return {
        "configurable": {
            "collection": None,
            "llm_key": None,
            "request_id": "live",
            "session_id": "live",
            "player_id": 1,
            "campaign_id": 1,
            "webhook": None,
        }
    }


def _context():
    return {
        "challenges": [
            {
                "id": 4378,
                "name": "Walk challenge",
                "linkToChallengeRules": [
                    {
                        "id": 4930,
                        "name": "Put on your walking shoes and go for a walk!",
                        "linkToDefaultGameDescriptor": {"id": 1, "translation_key": "WALK"},
                        "linkToRuleConditions": [
                            {
                                "linkToProperty": {"translation_key": "STEPS"},
                                "linkToOperator": {"operator": "STRICTLY_GREATER"},
                                "rhs_value": "499",
                            }
                        ],
                    }
                ],
            }
        ],
        "scheduledActivities": [],
        "schedulingWindow": {
            "scheduleStartDate": "2026-06-01",
            "scheduleEndDate": "2026-06-07",
            "schedulingPeriodWeeks": 1,
            "currentDate": "2026-06-01",
            "currentDayname": "mon",
        },
    }


def test_live_interviewer_returns_a_question():
    out = coach_graph.interview_node({}, _cfg())
    assert out["output"].strip()


def test_live_scheduling_pipeline_produces_a_verdict():
    state = {
        "messages": [
            {"role": "user", "content": "I wake up at 7 and walk every morning at 7am for 30 minutes."},
            {"role": "assistant", "content": "Thank you for your time. The interview is complete."},
        ],
        "context": _context(),
    }
    state.update(coach_graph.standardize_node(state, _cfg()))
    state.update(coach_graph.build_tasks_node(state, _cfg()))
    state.update(coach_graph.schedule_node(state, _cfg()))
    state.update(coach_graph.validate_node(state, _cfg()))
    state.update(coach_graph.finalize_node(state, _cfg()))

    assert isinstance(state["schedule_payload"].get("assignments"), list)
    assert isinstance(state["controller_valid"], bool)
    assert isinstance(state["planned_activities"], list)
