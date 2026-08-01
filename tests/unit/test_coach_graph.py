"""Unit tests for the LangGraph coach flow in src.agents.coach_graph."""

from __future__ import annotations

from datetime import date

from src.agents import coach_graph
from src.agents.ChromaDB.chroma_db import store_cleaned_conversation
from src.agents.coach_graph import (
    _append_messages,
    _assignment_to_planned_activity,
    _build_gamebus_task_index,
    _config_value,
    _emit,
    _is_interview_complete,
    _max_iterations,
    _parse_iso_date,
    _route_after_interview,
    _route_after_validate,
    _routines_as_busy_intervals,
    _store,
    _window_dates,
    build_coach_graph,
    build_tasks_node,
    clear_thread,
    finalize_node,
    interview_node,
    schedule_node,
    standardize_node,
    validate_node,
)


class FakeCollection:
    """Records stored documents and can fail on demand."""

    def __init__(self, fail: bool = False):
        self.docs = []
        self.fail = fail

    def add(self, documents, ids=None, metadatas=None):
        if self.fail:
            raise RuntimeError("add failed")
        self.docs.append(documents[0])


class FakeWebhook:
    """Captures emitted phase events."""

    def __init__(self):
        self.calls = []

    def emit(self, **kwargs):
        self.calls.append(kwargs)
        return True


def _cfg(**over):
    base = {
        "collection": None,
        "llm_key": None,
        "request_id": "r1",
        "session_id": "s1",
        "player_id": 1,
        "campaign_id": 2,
        "webhook": None,
    }
    base.update(over)
    return {"configurable": base}


def _context(challenges=None, scheduled=None, start="2026-06-01", end="2026-06-07"):
    return {
        "challenges": challenges or [],
        "scheduledActivities": scheduled or [],
        "schedulingWindow": {
            "scheduleStartDate": start,
            "scheduleEndDate": end,
            "schedulingPeriodWeeks": 1,
            "currentDate": start,
            "currentDayname": "mon",
        },
    }


def _walk_challenge():
    return {
        "id": 4378,
        "name": "Tasks",
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


# ---- small helpers --------------------------------------------------------------------------------


def test_max_iterations(monkeypatch):
    monkeypatch.delenv("COACH_MAX_SCHEDULE_ITERATIONS", raising=False)
    assert _max_iterations() == 3
    monkeypatch.setenv("COACH_MAX_SCHEDULE_ITERATIONS", "5")
    assert _max_iterations() == 5
    monkeypatch.setenv("COACH_MAX_SCHEDULE_ITERATIONS", "abc")
    assert _max_iterations() == 3


def test_parse_iso_date():
    assert _parse_iso_date("") is None
    assert _parse_iso_date("not-a-date") is None
    assert _parse_iso_date("2026-06-01") == date(2026, 6, 1)


def test_is_interview_complete():
    assert _is_interview_complete("the interview is complete now") is True
    assert _is_interview_complete("next question please") is False
    assert _is_interview_complete("") is False


def test_append_messages():
    assert _append_messages(None, None) == []
    assert _append_messages([1], [2]) == [1, 2]


def test_config_value():
    assert _config_value(None, "x", "d") == "d"
    assert _config_value({"configurable": {}}, "x", "d") == "d"
    assert _config_value({"configurable": {"x": 5}}, "x") == 5


def test_window_dates():
    start, end = _window_dates(
        {"schedulingWindow": {"scheduleStartDate": "2026-06-01", "scheduleEndDate": "2026-06-07"}}
    )
    assert start == date(2026, 6, 1)
    assert end == date(2026, 6, 7)
    assert _window_dates({}) == (None, None)


def test_emit_noop_without_webhook():
    assert _emit(_cfg(webhook=None), "INTERVIEWING") is None


def test_emit_calls_webhook():
    webhook = FakeWebhook()
    _emit(_cfg(webhook=webhook), "COMPLETE", percent=100, message="done")
    assert webhook.calls[0]["phase"] == "COMPLETE"
    assert webhook.calls[0]["percent_complete"] == 100


def test_store_noop_without_collection():
    assert _store(lambda *a, **k: None, None, "doc", "s", "r") is None


def test_store_success():
    collection = FakeCollection()
    _store(store_cleaned_conversation, collection, "doc", "s", "r")
    assert collection.docs == ["doc"]


def test_store_swallows_error():
    collection = FakeCollection(fail=True)
    _store(store_cleaned_conversation, collection, "doc", "s", "r")
    assert collection.docs == []


def test_routines_as_busy_intervals():
    out = _routines_as_busy_intervals(
        [
            {"date": "2026-06-01", "start": "07:00", "end": "07:30", "name": "wake"},
            "not-a-dict",
            {"date": "2026-06-01", "start": "07:00"},
        ]
    )
    assert len(out) == 1
    assert out[0]["source"] == "ROUTINE"
    assert out[0]["activityType"] == "wake"


def test_build_gamebus_task_index():
    assert _build_gamebus_task_index("nope") == {}
    task_sources = {
        "source_stores": {
            "gamebus_tasks": {
                "u1": {"task": {"task_id": 1, "game_descriptor_key": "WALK"}},
                "u2": {"task": "bad"},
                "u3": "noinfo",
            }
        }
    }
    index = _build_gamebus_task_index(task_sources)
    assert list(index.keys()) == ["u1"]


def test_assignment_to_planned_activity_defaults_and_flags():
    planned = _assignment_to_planned_activity(
        {"type": "gamebus", "date": "2026-06-01", "start": "09:00", "allDay": True},
        schedule_start=None,
        schedule_end=None,
    )
    assert planned.activityType == "ACTIVITY"
    assert planned.isHealthActivity is True
    assert planned.allDay is True
    assert planned.startDate == planned.endDate


# ---- nodes ----------------------------------------------------------------------------------------


def test_interview_node_first_turn_uses_start_prompt(monkeypatch):
    monkeypatch.setattr(
        "src.agents.interviewer_core.InterviewerAgent._send_to_llm", lambda self, messages: "Q1: hello"
    )
    out = interview_node({}, _cfg())
    assert out["messages"][0]["content"] == coach_graph.START_PROMPT
    assert out["messages"][1]["content"] == "Q1: hello"
    assert out["output"] == "Q1: hello"
    assert out["interview_complete"] is False


def test_interview_node_uses_user_message_after_first_turn(monkeypatch):
    monkeypatch.setattr(
        "src.agents.interviewer_core.InterviewerAgent._send_to_llm",
        lambda self, messages: "Thank you for your time.",
    )
    state = {
        "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        "message": "my answer",
    }
    out = interview_node(state, _cfg())
    assert out["messages"][0]["content"] == "my answer"
    assert out["interview_complete"] is True


def test_standardize_node_success(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(
        "src.agents.standardizer_core.StandardizerAgent.standardize",
        lambda self, message, clear_history=False: '{"Activities": []}',
    )
    state = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "there"}]}
    out = standardize_node(state, _cfg(collection=collection, llm_key="k"))
    assert out["standardized_json"] == '{"Activities": []}'
    assert out["cleaned_conversation"]
    assert len(collection.docs) == 2


def test_standardize_node_swallows_normalizer_failure(monkeypatch):
    monkeypatch.setattr(
        "src.agents.standardizer_core.StandardizerAgent.standardize",
        lambda self, message, clear_history=False: '{"Activities": []}',
    )

    def _boom(_payload):
        raise RuntimeError("normalizer down")

    monkeypatch.setattr("src.agents.coach_graph.normalize_standardized_activities", _boom)
    out = standardize_node({"messages": [{"role": "user", "content": "hi"}]}, _cfg())
    assert out["standardized_json"] == '{"Activities": []}'


def test_build_tasks_node_extracts_gamebus_and_expands_routines():
    collection = FakeCollection()
    standardized = (
        '{"Activities": [{"name": "walk routine", "frequency": {"type": "daily"}, '
        '"activity_duration": 30, "time": {"preferred": "07:00"}, '
        '"start_date": "2026-06-01", "IsInstantaneous": false}]}'
    )
    state = {"standardized_json": standardized, "context": _context(challenges=[_walk_challenge()])}
    out = build_tasks_node(state, _cfg(collection=collection))
    assert out["gamebus_task_count"] == 1
    assert out["iteration"] == 0
    assert out["controller_feedback"] == []
    assert out["routine_assignments"]
    assert out["gamebus_scheduler_sources"]["routine_tasks"] == []


def test_build_tasks_node_skips_expansion_when_window_unparseable():
    state = {"standardized_json": '{"Activities": []}', "context": _context(start="bad", end="bad")}
    out = build_tasks_node(state, _cfg())
    assert out["routine_assignments"] == []


def test_build_tasks_node_swallows_expander_failure(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("expander down")

    monkeypatch.setattr("src.agents.coach_graph.expand_routines_to_assignments", _boom)
    standardized = (
        '{"Activities": [{"name": "x", "frequency": {"type": "daily"}, '
        '"time": {"preferred": "07:00"}, "start_date": "2026-06-01"}]}'
    )
    out = build_tasks_node({"standardized_json": standardized, "context": _context()}, _cfg())
    assert out["routine_assignments"] == []


def test_schedule_node_uses_schedule_json(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(
        "src.agents.coach_graph.schedule_from_inputs",
        lambda **k: {
            "schedule_json": {
                "assignments": [{"task_id": "g1", "date": "2026-06-01", "start": "09:00", "end": "10:00"}]
            }
        },
    )
    out = schedule_node(
        {"context": _context(), "gamebus_scheduler_sources": {}, "iteration": 0}, _cfg(collection=collection)
    )
    assert out["schedule_payload"]["assignments"]
    assert out["iteration"] == 1
    assert len(collection.docs) == 1


def test_schedule_node_falls_back_to_assignments_key(monkeypatch):
    monkeypatch.setattr(
        "src.agents.coach_graph.schedule_from_inputs",
        lambda **k: {"assignments": [{"task_id": "g1", "date": "2026-06-01", "start": "09:00"}]},
    )
    out = schedule_node({"context": _context(), "iteration": 1}, _cfg())
    assert out["schedule_payload"]["assignments"]
    assert out["iteration"] == 2


def test_schedule_node_handles_non_dict_result(monkeypatch):
    monkeypatch.setattr("src.agents.coach_graph.schedule_from_inputs", lambda **k: "nope")
    out = schedule_node({"context": _context(), "iteration": 0}, _cfg())
    assert out["schedule_payload"] == {"assignments": []}


def test_schedule_node_swallows_runtime_error(monkeypatch):
    def _raise(**_k):
        raise RuntimeError("no tasks")

    monkeypatch.setattr("src.agents.coach_graph.schedule_from_inputs", _raise)
    out = schedule_node({"context": _context(), "iteration": 0}, _cfg())
    assert out["schedule_payload"] == {"assignments": []}
    assert out["iteration"] == 1


def test_validate_node_skips_controller_when_no_assignments():
    out = validate_node({"schedule_payload": {"assignments": []}}, _cfg())
    assert out["controller_valid"] is True
    assert out["controller_feedback"] == []


def test_validate_node_collects_feedback_when_invalid(monkeypatch):
    monkeypatch.setattr(
        "src.agents.controller_core.ControllerAgent.validate_schedule",
        lambda self, **k: {"is_valid": False, "violations": ["v1"], "fix_advice": ["f1"]},
    )
    out = validate_node(
        {
            "schedule_payload": {"assignments": [{"x": 1}]},
            "gamebus_scheduler_sources": {},
            "gamebus_task_count": 1,
        },
        _cfg(),
    )
    assert out["controller_valid"] is False
    assert out["controller_feedback"] == ["v1", "f1"]


def test_validate_node_swallows_controller_error(monkeypatch):
    def _raise(self, **_k):
        raise RuntimeError("controller down")

    monkeypatch.setattr("src.agents.controller_core.ControllerAgent.validate_schedule", _raise)
    out = validate_node({"schedule_payload": {"assignments": [{"x": 1}]}}, _cfg())
    assert out["controller_valid"] is True
    assert out["controller_feedback"] == []


def test_finalize_node_links_gamebus_and_filters_window():
    webhook = FakeWebhook()
    task_sources = {
        "source_stores": {"gamebus_tasks": {"u1": {"task": {"task_id": 4930, "game_descriptor_key": "WALK"}}}}
    }
    state = {
        "context": _context(),
        "routine_assignments": [],
        "schedule_payload": {
            "assignments": [
                {"task_id": "u1", "name": "raw", "date": "2026-06-01", "start": "09:00", "end": "10:00"},
                {"task_id": "u2", "name": "out", "date": "2026-12-01", "start": "09:00", "end": "10:00"},
            ]
        },
        "task_sources": task_sources,
    }
    out = finalize_node(state, _cfg(webhook=webhook))
    planned = out["planned_activities"]
    assert len(planned) == 1
    assert planned[0].activityType == "WALK"
    assert planned[0].challengeRuleId == 4930
    assert webhook.calls[0]["phase"] == "COMPLETE"


def test_finalize_node_gamebus_source_without_ids():
    task_sources = {
        "source_stores": {"gamebus_tasks": {"u1": {"task": {"task_id": None, "game_descriptor_key": None}}}}
    }
    state = {
        "context": _context(),
        "routine_assignments": [
            {"task_id": "u1", "name": "r", "date": "2026-06-01", "start": "09:00", "end": "10:00"}
        ],
        "schedule_payload": {"assignments": []},
        "task_sources": task_sources,
    }
    out = finalize_node(state, _cfg())
    assert out["planned_activities"][0].activityType == "r"


def test_finalize_node_skips_non_dict_merged(monkeypatch):
    monkeypatch.setattr(
        "src.agents.coach_graph.merge_assignments",
        lambda r, s: ["skip", {"task_id": "x", "date": "2026-06-01", "start": "09:00", "end": "10:00"}],
    )
    state = {
        "context": _context(),
        "routine_assignments": [],
        "schedule_payload": {"assignments": []},
        "task_sources": {},
    }
    out = finalize_node(state, _cfg())
    assert len(out["planned_activities"]) == 1


# ---- routing --------------------------------------------------------------------------------------


def test_route_after_interview():
    assert _route_after_interview({"interview_complete": True}) == "standardize"
    assert _route_after_interview({"interview_complete": False}) == coach_graph.END


def test_route_after_validate(monkeypatch):
    assert _route_after_validate({"controller_valid": True, "iteration": 1}) == "finalize"
    monkeypatch.setenv("COACH_MAX_SCHEDULE_ITERATIONS", "2")
    assert _route_after_validate({"controller_valid": False, "iteration": 2}) == "finalize"
    assert _route_after_validate({"controller_valid": False, "iteration": 1}) == "schedule"


# ---- graph build, reset, and the full loop --------------------------------------------------------


def test_build_coach_graph_compiles():
    assert build_coach_graph() is not None


def test_clear_thread_pops_storage():
    graph = build_coach_graph()
    graph.checkpointer.storage["t1"] = {"x": 1}
    clear_thread(graph, "t1")
    assert "t1" not in graph.checkpointer.storage


def test_clear_thread_handles_missing_storage():
    class _NoSaver:
        pass

    clear_thread(_NoSaver(), "t1")


def _graph_config(thread_id, **over):
    base = {
        "thread_id": thread_id,
        "collection": None,
        "llm_key": None,
        "request_id": "r",
        "session_id": "s",
        "player_id": 1,
        "campaign_id": 2,
        "webhook": None,
    }
    base.update(over)
    return {"configurable": base}


def test_graph_stops_after_incomplete_interview(monkeypatch):
    monkeypatch.setattr("src.agents.interviewer_core.InterviewerAgent._send_to_llm", lambda self, m: "Q1: hi")
    graph = build_coach_graph()
    out = graph.invoke({"message": "", "context": _context()}, _graph_config("t-inc"))
    assert out["interview_complete"] is False
    assert "planned_activities" not in out


def test_graph_runs_controller_loop_until_valid(monkeypatch):
    monkeypatch.setattr(
        "src.agents.interviewer_core.InterviewerAgent._send_to_llm",
        lambda self, m: "Thank you for your time.",
    )
    monkeypatch.setattr(
        "src.agents.standardizer_core.StandardizerAgent.standardize",
        lambda self, message, clear_history=False: '{"Activities": []}',
    )
    calls = {"schedule": 0, "validate": 0}

    def _sched(**_k):
        calls["schedule"] += 1
        return {
            "schedule_json": {
                "assignments": [
                    {"task_id": "g1", "name": "WALK", "date": "2026-06-01", "start": "09:00", "end": "10:00"}
                ]
            }
        }

    monkeypatch.setattr("src.agents.coach_graph.schedule_from_inputs", _sched)

    def _validate(self, **_k):
        calls["validate"] += 1
        return {"is_valid": calls["validate"] >= 2, "violations": ["v"], "fix_advice": ["f"]}

    monkeypatch.setattr("src.agents.controller_core.ControllerAgent.validate_schedule", _validate)

    graph = build_coach_graph()
    out = graph.invoke({"message": "go", "context": _context()}, _graph_config("t-loop"))
    assert out["interview_complete"] is True
    assert calls["schedule"] == 2
    assert out["planned_activities"]


def test_graph_stops_at_max_iterations(monkeypatch):
    monkeypatch.setenv("COACH_MAX_SCHEDULE_ITERATIONS", "2")
    monkeypatch.setattr(
        "src.agents.interviewer_core.InterviewerAgent._send_to_llm", lambda self, m: "Interview completed."
    )
    monkeypatch.setattr(
        "src.agents.standardizer_core.StandardizerAgent.standardize",
        lambda self, message, clear_history=False: '{"Activities": []}',
    )
    calls = {"schedule": 0}

    def _sched(**_k):
        calls["schedule"] += 1
        return {
            "schedule_json": {
                "assignments": [
                    {"task_id": "g1", "name": "WALK", "date": "2026-06-01", "start": "09:00", "end": "10:00"}
                ]
            }
        }

    monkeypatch.setattr("src.agents.coach_graph.schedule_from_inputs", _sched)
    monkeypatch.setattr(
        "src.agents.controller_core.ControllerAgent.validate_schedule",
        lambda self, **_k: {"is_valid": False, "violations": ["v"], "fix_advice": ["f"]},
    )

    graph = build_coach_graph()
    out = graph.invoke({"message": "go", "context": _context()}, _graph_config("t-max"))
    assert calls["schedule"] == 2
    assert out["interview_complete"] is True
