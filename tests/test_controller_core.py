from src.agents.controller_core import ControllerAgent


def test_controller_returns_valid_when_llm_disabled_routine_only():
    controller = ControllerAgent(enable_llm=False)
    result = controller.validate_schedule(
        schedule_payload={"assignments": [{"type": "routine", "task_id": "r1", "date": "2026-02-25"}]},
        expected_gamebus_tasks=2,
    )
    assert result["is_valid"] is True
    assert result["violations"] == []


def test_controller_returns_valid_when_llm_disabled_gamebus_present():
    controller = ControllerAgent(enable_llm=False)
    result = controller.validate_schedule(
        schedule_payload={
            "assignments": [
                {"type": "gamebus", "task_id": "g1", "date": "2026-02-25", "start": "09:00", "end": "09:15"}
            ]
        },
        expected_gamebus_tasks=1,
    )
    assert result["is_valid"] is True
    assert result["violations"] == []


def test_controller_summary_counts_assignments():
    controller = ControllerAgent(enable_llm=False)
    result = controller.validate_schedule(
        schedule_payload={
            "assignments": [
                {"type": "routine", "task_id": "r1", "date": "2026-02-25"},
                {"type": "gamebus", "task_id": "g1", "date": "2026-02-26", "start": "09:00", "end": "09:15"},
            ]
        },
        expected_gamebus_tasks=1,
    )
    assert result["summary"]["assignments"] == 2
    assert result["summary"]["expected_gamebus_tasks"] == 1
