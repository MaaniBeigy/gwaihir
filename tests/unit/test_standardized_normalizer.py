"""Correction of `frequency.days` from the standardizer description text."""

from src.agents.tools.standardized_normalizer import normalize_standardized_activities


def _workday_routine(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "frequency": {"type": "daily", "days": None, "count": 1},
        "activity_duration": 30,
        "start_date": "2026-05-16",
        "time": {"preferred": "08:00"},
    }


def test_workdays_phrase_in_description_corrects_to_mon_fri() -> None:
    payload = {
        "Activities": [
            _workday_routine("work", "I usually start working at 08:00 on workdays."),
            _workday_routine("eat lunch", "I usually eat lunch at 12:00 for 30 minutes on workdays."),
            _workday_routine("commute to work", "I cycle to work on workdays at 07:30."),
        ],
    }
    out = normalize_standardized_activities(payload)
    for a in out["Activities"]:
        assert a["frequency"]["type"] == "specific_days", a["name"]
        assert a["frequency"]["days"] == ["mon", "tue", "wed", "thu", "fri"], a["name"]


def test_weekend_phrase_corrects_to_sat_sun() -> None:
    payload = {
        "Activities": [
            _workday_routine("brunch", "We have brunch on weekends at 11:00."),
        ],
    }
    out = normalize_standardized_activities(payload)
    assert out["Activities"][0]["frequency"]["days"] == ["sat", "sun"]


def test_explicit_days_in_description_override_daily() -> None:
    payload = {
        "Activities": [
            _workday_routine("gym", "I go to the gym on Tuesday and Thursday at 19:30 for 2 hours."),
        ],
    }
    out = normalize_standardized_activities(payload)
    assert out["Activities"][0]["frequency"]["days"] == ["tue", "thu"]


def test_specific_days_with_explicit_days_is_left_alone() -> None:
    """Trust `specific_days` plus an explicit `days` list."""
    payload = {
        "Activities": [
            {
                "name": "gym",
                "description": "Saturdays at the gym.",
                "frequency": {"type": "specific_days", "days": ["sat"], "count": 1},
                "activity_duration": 60,
                "start_date": "2026-05-16",
                "time": {"preferred": "16:30"},
            }
        ],
    }
    out = normalize_standardized_activities(payload)
    assert out["Activities"][0]["frequency"]["days"] == ["sat"]


def test_truly_daily_routine_with_no_signal_is_left_alone() -> None:
    """A `daily` frequency stays daily when the description has no day signal."""
    payload = {
        "Activities": [
            {
                "name": "watch movies",
                "description": "I watch movies every night at 22:00 for 45 minutes.",
                "frequency": {"type": "daily", "days": None, "count": 1},
                "activity_duration": 45,
                "start_date": "2026-05-16",
                "time": {"preferred": "22:00"},
            }
        ],
    }
    out = normalize_standardized_activities(payload)
    assert out["Activities"][0]["frequency"]["type"] == "daily"
    assert out["Activities"][0]["frequency"]["days"] is None


def test_string_input_returns_string_with_corrections() -> None:
    import json

    payload = json.dumps(
        {
            "Activities": [_workday_routine("work", "I work on workdays from 08:00 to 17:00.")],
        }
    )
    out = normalize_standardized_activities(payload)
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert parsed["Activities"][0]["frequency"]["days"] == ["mon", "tue", "wed", "thu", "fri"]


def test_none_input_returns_none() -> None:
    assert normalize_standardized_activities(None) is None


def test_incomplete_workday_specific_days_is_expanded() -> None:
    """A work routine collapsed to ["mon","fri"] is restored to the full work week."""
    payload = {
        "Activities": [
            {
                "name": "work",
                "description": "I usually start working at 08:00 and finish at 17:00 on workdays.",
                "frequency": {"type": "specific_days", "days": ["mon", "fri"], "count": 1},
                "activity_duration": 540,
                "start_date": "2026-06-01",
                "time": {"preferred": "08:00"},
                "notes": "workday schedule applies Monday to Friday",
            }
        ],
    }
    out = normalize_standardized_activities(payload)
    assert out["Activities"][0]["frequency"]["days"] == ["mon", "tue", "wed", "thu", "fri"]


def test_incomplete_weekend_specific_days_is_expanded() -> None:
    """A weekend routine with only one day is restored to sat and sun."""
    payload = {
        "Activities": [
            {
                "name": "brunch",
                "description": "We have brunch on weekends.",
                "frequency": {"type": "specific_days", "days": ["sat"], "count": 1},
                "activity_duration": 60,
                "start_date": "2026-06-06",
                "time": {"preferred": "11:00"},
            }
        ],
    }
    out = normalize_standardized_activities(payload)
    assert out["Activities"][0]["frequency"]["days"] == ["sat", "sun"]


def test_monday_to_friday_range_not_collapsed_to_endpoints() -> None:
    """A "Monday to Friday" range is read as five days, not just its endpoints."""
    payload = {
        "Activities": [
            {
                "name": "work",
                "description": "I work Monday to Friday from 08:00 to 17:00.",
                "frequency": {"type": "daily", "days": None, "count": 1},
                "activity_duration": 540,
                "start_date": "2026-06-01",
                "time": {"preferred": "08:00"},
            }
        ],
    }
    out = normalize_standardized_activities(payload)
    assert out["Activities"][0]["frequency"]["days"] == ["mon", "tue", "wed", "thu", "fri"]


def test_complete_workday_set_with_phrase_is_left_alone() -> None:
    """A complete work week is not rewritten when a workday phrase is present."""
    payload = {
        "Activities": [
            {
                "name": "work",
                "description": "I work on workdays.",
                "frequency": {
                    "type": "specific_days",
                    "days": ["mon", "tue", "wed", "thu", "fri"],
                    "count": 3,
                },
                "activity_duration": 540,
                "start_date": "2026-06-01",
                "time": {"preferred": "08:00"},
            }
        ],
    }
    out = normalize_standardized_activities(payload)
    assert out["Activities"][0]["frequency"]["days"] == ["mon", "tue", "wed", "thu", "fri"]
    # count untouched proves the activity was not re-written.
    assert out["Activities"][0]["frequency"]["count"] == 3


def test_complete_weekend_set_with_phrase_is_left_alone() -> None:
    """A complete weekend is not rewritten when a weekend phrase is present."""
    payload = {
        "Activities": [
            {
                "name": "brunch",
                "description": "We have brunch on weekends.",
                "frequency": {"type": "specific_days", "days": ["sat", "sun"], "count": 4},
                "activity_duration": 60,
                "start_date": "2026-06-06",
                "time": {"preferred": "11:00"},
            }
        ],
    }
    out = normalize_standardized_activities(payload)
    assert out["Activities"][0]["frequency"]["days"] == ["sat", "sun"]
    assert out["Activities"][0]["frequency"]["count"] == 4
