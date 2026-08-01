"""Unit tests for `src.agents.tools.gamebus_task_extractor`."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.agents.tools.gamebus_task_extractor import (
    extract_task,
    flatten_condition,
    process_json,
    simplify_challenge,
)

# ---- flatten_condition -------------------------------------------------------------------------


def test_flatten_condition_extracts_property_operator_value():
    condition = {
        "linkToProperty": {"translation_key": "DURATION"},
        "linkToOperator": {"operator": "STRICTLY_GREATER"},
        "rhs_value": 600,
    }

    flat = flatten_condition(condition)

    assert flat == {
        "property": "DURATION",
        "operator": "STRICTLY_GREATER",
        "value": 600,
    }


def test_flatten_condition_falls_back_to_unknown_when_fields_missing():
    flat = flatten_condition({})

    assert flat == {
        "property": "unknown",
        "operator": "unknown",
        "value": "unknown",
    }


# ---- extract_task: condition branches ----------------------------------------------------------


def _rule_with_conditions(conditions, *, name="Walk", description=""):
    return {
        "id": 42,
        "name": name,
        "description": description,
        "linkToDefaultGameDescriptor": {"id": 7, "translation_key": "WALK"},
        "linkToPointMappings": [{"points": 10}],
        "linkToRuleConditions": conditions,
    }


def test_extract_task_uses_duration_condition_seconds_to_minutes():
    """DURATION in seconds becomes minutes plus a 5-minute buffer."""
    rule = _rule_with_conditions(
        [
            {
                "linkToProperty": {"translation_key": "DURATION"},
                "linkToOperator": {"operator": "STRICTLY_GREATER"},
                "rhs_value": 600,
            }
        ]
    )

    task = extract_task(rule)

    # 600 / 60 = 10 minutes, plus 5 minute buffer.
    assert task["est_duration_min"] == 15
    assert task["min_steps"] is None
    assert task["task_id"] == 42
    assert task["game_descriptor_key"] == "WALK"
    assert task["points"] == 10
    assert task["conditions"][0]["property"] == "DURATION"


def test_extract_task_uses_steps_condition_to_derive_duration():
    """STEPS becomes `min_steps` plus a derived duration at about 110 steps per minute."""
    rule = _rule_with_conditions(
        [
            {
                "linkToProperty": {"translation_key": "STEPS"},
                "linkToOperator": {"operator": "STRICTLY_GREATER"},
                "rhs_value": 1000,
            }
        ]
    )

    task = extract_task(rule)

    expected_min_steps = 1001
    expected_duration = math.ceil(expected_min_steps / 110) + 5
    assert task["min_steps"] == expected_min_steps
    assert task["est_duration_min"] == expected_duration


def test_extract_task_uses_steps_sum_condition_for_daily_aggregate():
    """STEPS_SUM (e.g. 5000 steps per day) takes the same derivation path as STEPS."""
    rule = _rule_with_conditions(
        [
            {
                "linkToProperty": {"translation_key": "STEPS_SUM"},
                "linkToOperator": {"operator": "STRICTLY_GREATER"},
                "rhs_value": 4999,
            }
        ]
    )

    task = extract_task(rule)

    assert task["min_steps"] == 5000
    assert task["est_duration_min"] == math.ceil(5000 / 110) + 5


def test_extract_task_combined_conditions_keep_max_duration():
    """Combining DURATION and STEPS_SUM keeps the larger derived duration."""
    rule = _rule_with_conditions(
        [
            {
                "linkToProperty": {"translation_key": "DURATION"},
                "linkToOperator": {"operator": "STRICTLY_GREATER"},
                "rhs_value": 60,
            },
            {
                "linkToProperty": {"translation_key": "STEPS_SUM"},
                "linkToOperator": {"operator": "STRICTLY_GREATER"},
                "rhs_value": 9999,
            },
        ]
    )

    task = extract_task(rule)

    assert task["min_steps"] == 10000
    assert task["est_duration_min"] == math.ceil(10000 / 110) + 5


def test_extract_task_unrelated_condition_does_not_set_duration():
    """A condition that is neither DURATION nor STEPS leaves duration to the heuristics."""
    rule = _rule_with_conditions(
        [
            {
                "linkToProperty": {"translation_key": "POINTS"},
                "linkToOperator": {"operator": "STRICTLY_GREATER"},
                "rhs_value": 5,
            }
        ],
        name="Untagged activity",
        description="Nothing duration-like in the description",
    )

    task = extract_task(rule)

    # Falls through to the GENERAL_ACTIVITY default of 5 minutes.
    assert task["est_duration_min"] == 5
    assert task["min_steps"] is None


# ---- extract_task: text heuristics for default duration estimates --------------------------------


@pytest.mark.parametrize(
    "name,description,expected_minutes",
    [
        # 15-minute family: flat 20.
        ("Take a 15-minute walk", "", 20),
        ("Take a 15 min walk", "", 20),
        ("Take a 15 minutes walk", "", 20),
        # 30-minute family: 35 when the literal "30 m" appears, else 2.
        ("Take a 30 min run", "", 35),
        ("30 minutes of yoga", "", 35),
        ("Daily 30-minute run", "", 2),
        ("Hold the yoga pose", "30 seconds is enough", 2),
        ("Yoga pose", "Hold for a moment", 2),
        # 7-minute and workout family: 10.
        ("7-minute workout", "", 10),
        ("H5P quiz", "", 10),
        ("Generic workout", "", 10),
        # photo, picture, image: 3.
        ("Upload a photo", "", 3),
        ("Send a picture", "", 3),
        ("Capture an image", "", 3),
        # Default: 5.
        ("Random activity", "Nothing special", 5),
    ],
)
def test_extract_task_default_duration_text_heuristics(name, description, expected_minutes):
    rule = _rule_with_conditions([], name=name, description=description)

    task = extract_task(rule)

    assert task["est_duration_min"] == expected_minutes


# ---- simplify_challenge ------------------------------------------------------------------------


def test_simplify_challenge_keeps_named_rules_only():
    challenge = {
        "id": 1,
        "name": "Walking Challenge",
        "description": "Walk every day",
        "challenge_type": "TASKS_COLLECTION",
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "linkToChallengeRules": [
            {
                "id": 11,
                "name": "Daily walk",
                "description": "Take a 30-minute walk",
                "linkToDefaultGameDescriptor": {"id": 1, "translation_key": "WALK"},
                "linkToPointMappings": [{"points": 5}],
                "linkToRuleConditions": [],
            },
            {
                # Unnamed rule must be filtered out.
                "id": 12,
                "name": "",
                "linkToChallengeRules": [],
            },
        ],
    }

    simplified = simplify_challenge(challenge)

    assert simplified["challenge_id"] == 1
    assert simplified["type"] == "TASKS_COLLECTION"
    assert len(simplified["tasks"]) == 1
    assert simplified["tasks"][0]["task_id"] == 11


# ---- process_json ------------------------------------------------------------------------------


def test_process_json_round_trip(tmp_path: Path, capsys):
    input_path = tmp_path / "challenges.json"
    output_path = tmp_path / "slim.json"

    input_path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "name": "Walking Challenge",
                    "description": "Walk every day",
                    "challenge_type": "TASKS_COLLECTION",
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-30",
                    "linkToChallengeRules": [
                        {
                            "id": 11,
                            "name": "Daily walk",
                            "description": "Take a 30-minute walk",
                            "linkToDefaultGameDescriptor": {
                                "id": 1,
                                "translation_key": "WALK",
                            },
                            "linkToPointMappings": [{"points": 5}],
                            "linkToRuleConditions": [],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    process_json(str(input_path), str(output_path))

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["total_challenges"] == 1
    assert payload["challenges"][0]["challenge_id"] == 1
    assert payload["challenges"][0]["tasks"][0]["task_id"] == 11

    captured = capsys.readouterr()
    assert "Processed 1 challenges" in captured.out
    assert "Example first task" in captured.out


def test_process_json_handles_challenge_without_tasks(tmp_path: Path, capsys):
    """When the first challenge has zero rules, the example-print branch is skipped."""
    input_path = tmp_path / "empty.json"
    output_path = tmp_path / "slim_empty.json"

    input_path.write_text(
        json.dumps(
            [
                {
                    "id": 9,
                    "name": "Empty",
                    "description": "",
                    "challenge_type": "TASKS_COLLECTION",
                    "start_date": "2026-04-01",
                    "linkToChallengeRules": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    process_json(str(input_path), str(output_path))

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["challenges"][0]["tasks"] == []

    captured = capsys.readouterr()
    assert "Processed 1 challenges" in captured.out
    assert "Example first task" not in captured.out
