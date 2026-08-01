from datetime import date

from src.agents.tools.task_normalizer import build_unified_scheduler_tasks


def test_build_unified_scheduler_tasks_accepts_raw_gamebus_challenge_list():
    gamebus_payload = [
        {
            "id": 101,
            "challenge_type": "TASKS_COLLECTION",
            "name": "Challenge Raw",
            "start_date": "2026-02-01",
            "end_date": "2026-02-28",
            "linkToChallengeRules": [
                {
                    "id": 501,
                    "name": "Take a short walk",
                    "description": "Walk for 15 minutes",
                    "linkToDefaultGameDescriptor": {"id": 1, "translation_key": "GENERAL_ACTIVITY"},
                    "linkToPointMappings": [{"points": 5}],
                    "linkToRuleConditions": [],
                }
            ],
        }
    ]

    result = build_unified_scheduler_tasks(gamebus_payload=gamebus_payload)

    assert result["gamebus_tasks"] == 1
    assert result["total_tasks"] == 1
    assert result["tasks"][0]["source_type"] == "gamebus"


def test_build_unified_scheduler_tasks_accepts_simplified_gamebus_envelope():
    gamebus_payload = {
        "simplified_at": "2026-02-26T09:00:00",
        "total_challenges": 1,
        "challenges": [
            {
                "challenge_id": 201,
                "name": "Challenge Simplified",
                "description": "Already simplified",
                "type": "TASKS_COLLECTION",
                "start_date": "2026-02-01",
                "end_date": "2026-02-28",
                "tasks": [
                    {
                        "task_id": 601,
                        "name": "Upload photo",
                        "description": "Take and upload a photo",
                        "est_duration_min": 3,
                        "max_repeats": 10,
                        "cooldown_days": 7,
                    }
                ],
            }
        ],
    }

    result = build_unified_scheduler_tasks(gamebus_payload=gamebus_payload)

    assert result["gamebus_tasks"] == 1
    task = result["tasks"][0]
    assert "task_uuid" not in task
    assert task["source_type"] == "gamebus"
    assert task["name"] == "Upload photo"
    assert task["description"] == "Take and upload a photo"
    assert task["est_duration_min"] == 3
    assert task["max_repeats"] == 10
    assert task["cooldown_days"] == 7


def test_build_unified_scheduler_tasks_ignores_invalid_gamebus_payload_without_crashing():
    result = build_unified_scheduler_tasks(gamebus_payload={"unexpected": "shape"})

    assert result["gamebus_tasks"] == 0
    assert result["total_tasks"] == 0
    assert result["tasks"] == []


def test_build_unified_scheduler_tasks_expands_fixed_specific_days_routines_to_concrete_dates():
    standardized_payload = {
        "Activities": [
            {
                "name": "eat breakfast",
                "description": "Weekday breakfast",
                "IsInstantaneous": False,
                "frequency": {
                    "type": "specific_days",
                    "days": ["mon", "tue", "wed", "thu", "fri"],
                    "count": 1,
                },
                "activity_duration": 10,
                "start_date": "2026-03-09",
                "time": {"preferred": "07:00"},
                "notes": "weekday routine",
            }
        ]
    }

    result = build_unified_scheduler_tasks(
        standardized_activities_payload=standardized_payload,
        schedule_start=date(2026, 3, 9),
        schedule_end=date(2026, 3, 15),
    )

    routine_tasks = [task for task in result["tasks"] if task.get("source_type") == "routine"]
    assert len(routine_tasks) == 5

    expected_dates = ["2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13"]
    assert sorted(task["start_date"] for task in routine_tasks) == expected_dates
    assert all(task.get("end_date") == task.get("start_date") for task in routine_tasks)
    assert all(task.get("routine_schedule_mode") == "fixed" for task in routine_tasks)
    assert all(task.get("scheduling_required") is False for task in routine_tasks)
    assert all((task.get("frequency") or {}).get("count") == 1 for task in routine_tasks)
    assert all(task.get("preferred_start_time") == "07:00" for task in routine_tasks)
    assert all(task.get("preferred_end_time") == "07:10" for task in routine_tasks)


def test_build_unified_scheduler_tasks_preserves_end_date_for_window_routines():
    standardized_payload = {
        "Activities": [
            {
                "name": "read",
                "description": "Read in the evening",
                "IsInstantaneous": False,
                "frequency": {
                    "type": "specific_days",
                    "days": ["mon"],
                    "count": 1,
                },
                "activity_duration": 30,
                "start_date": "2026-03-09",
                "end_date": "2026-03-30",
                "time": {"preferred": "evening"},
                "notes": None,
            }
        ]
    }

    result = build_unified_scheduler_tasks(
        standardized_activities_payload=standardized_payload,
        schedule_start=date(2026, 3, 9),
        schedule_end=date(2026, 4, 5),
    )
    routine_tasks = [task for task in result["tasks"] if task.get("source_type") == "routine"]

    assert len(routine_tasks) == 1
    task = routine_tasks[0]
    assert task["routine_schedule_mode"] == "window"
    assert task["scheduling_required"] is True
    assert task["start_date"] == "2026-03-09"
    assert task["end_date"] == "2026-03-30"
    assert task["preferred_start_time"] is None
    assert task["preferred_end_time"] is None


def test_fixed_specific_days_expansion_uses_scheduling_period_weeks(monkeypatch):
    monkeypatch.setenv("SCHEDULING_PERIOD_WEEKS", "2")

    standardized_payload = {
        "Activities": [
            {
                "name": "weekday breakfast",
                "description": "Breakfast on weekdays",
                "IsInstantaneous": False,
                "frequency": {
                    "type": "specific_days",
                    "days": ["mon", "tue", "wed", "thu", "fri"],
                    "count": 1,
                },
                "activity_duration": 10,
                "start_date": "2026-03-09",
                "time": {"preferred": "07:00"},
            }
        ]
    }

    # Pin schedule_start; schedule_end falls back to start + (period_weeks*7 - 1).
    result = build_unified_scheduler_tasks(
        standardized_activities_payload=standardized_payload,
        schedule_start=date(2026, 3, 9),
    )
    routine_tasks = [task for task in result["tasks"] if task.get("source_type") == "routine"]

    # Two-week horizon from 2026-03-09 includes 10 weekdays.
    assert len(routine_tasks) == 10
    assert min(task["start_date"] for task in routine_tasks) == "2026-03-09"
    assert max(task["start_date"] for task in routine_tasks) == "2026-03-20"
