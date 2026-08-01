"""Conversion of scheduler assignments to plannedActivities."""

from datetime import date

from src.agents.app import _assignment_to_planned_activity, _build_gamebus_task_index
from src.agents.tools.task_normalizer import build_unified_scheduler_tasks


def test_assignment_inside_window_is_kept():
    planned = _assignment_to_planned_activity(
        {
            "type": "gamebus",
            "task_id": "g1",
            "name": "WALK",
            "date": "2026-04-17",
            "start": "09:00",
            "end": "10:00",
            "challengeRuleId": 123,
        },
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 19),
    )
    assert planned is not None
    # Gamebus stores wall-clock values without timezone, so the emitted ISO has no Z or offset.
    assert planned.startDate == "2026-04-17T09:00:00"
    assert planned.endDate == "2026-04-17T10:00:00"
    assert "Z" not in planned.startDate
    assert "+" not in planned.startDate
    assert planned.challengeRuleId == 123
    assert planned.gameDescriptorTK == "SCHEDULE_ACTIVITY"


def test_assignment_outside_window_is_dropped():
    planned = _assignment_to_planned_activity(
        {
            "type": "gamebus",
            "task_id": "g1",
            "name": "WALK",
            "date": "2026-04-25",
            "start": "09:00",
            "end": "10:00",
        },
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 19),
    )
    assert planned is None


def test_assignment_without_date_is_dropped():
    planned = _assignment_to_planned_activity(
        {"name": "WALK", "start": "09:00", "end": "10:00"},
        schedule_start=None,
        schedule_end=None,
    )
    assert planned is None


def test_routine_assignment_uses_task_id_when_no_challenge_rule_id():
    planned = _assignment_to_planned_activity(
        {
            "type": "routine",
            "task_id": "55",
            "name": "Breakfast",
            "date": "2026-04-17",
            "start": "08:00",
            "end": "08:15",
        },
        schedule_start=None,
        schedule_end=None,
    )
    assert planned is not None
    assert planned.challengeRuleId == 55
    assert planned.activityType == "Breakfast"


def test_planned_activity_prefers_activity_type_over_name():
    # The coach sets activityType to the game descriptor key; it must win over the rule name.
    planned = _assignment_to_planned_activity(
        {
            "type": "gamebus",
            "task_id": "g1",
            "name": "Put on your walking shoes!",
            "activityType": "WALK",
            "challengeRuleId": 4930,
            "date": "2026-06-05",
            "start": "21:00",
            "end": "21:10",
        },
        schedule_start=date(2026, 6, 5),
        schedule_end=date(2026, 6, 11),
    )
    assert planned is not None
    assert planned.activityType == "WALK"
    assert planned.challengeRuleId == 4930


def test_gamebus_task_index_exposes_descriptor_for_any_challenge():
    challenge = {
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
                    },
                ],
            }
        ],
    }
    task_sources = build_unified_scheduler_tasks(
        standardized_activities_payload={"Activities": []},
        gamebus_payload={"challenges": [challenge]},
        schedule_start=date(2026, 6, 5),
        schedule_end=date(2026, 6, 11),
    )
    index = _build_gamebus_task_index(task_sources)
    assert len(index) == 1
    task = next(iter(index.values()))
    # The coach maps activityType to this descriptor key so the studio calendar recognizes the
    # slot as a challenge task and routes completion through the normal scoring path.
    assert task["game_descriptor_key"] == "WALK"
    assert task["task_id"] == 4930


def test_assignment_to_planned_activity_drops_when_missing_start():
    planned = _assignment_to_planned_activity({"date": "2026-04-15"}, schedule_start=None, schedule_end=None)
    assert planned is None


def test_assignment_to_planned_activity_drops_when_before_window():
    planned = _assignment_to_planned_activity(
        {"date": "2026-04-01", "start": "09:00"},
        schedule_start=date(2026, 4, 13),
        schedule_end=date(2026, 4, 19),
    )
    assert planned is None


def test_assignment_to_planned_activity_handles_non_castable_challenge_rule():
    planned = _assignment_to_planned_activity(
        {"date": "2026-04-15", "start": "09:00", "challengeRuleId": "not-an-int", "type": "gamebus"},
        schedule_start=None,
        schedule_end=None,
    )
    assert planned is not None
    assert planned.challengeRuleId is None
