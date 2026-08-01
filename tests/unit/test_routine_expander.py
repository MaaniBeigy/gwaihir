"""Expansion of standardized routines into one assignment per occurrence."""

from datetime import date

from src.agents.tools.routine_expander import expand_routines_to_assignments, merge_assignments


def _wake_up_workdays() -> dict:
    return {
        "name": "wake up",
        "description": "On workdays I wake up at 07:00.",
        "IsInstantaneous": True,
        "IsRegular": True,
        "frequency": {"type": "specific_days", "days": ["mon", "tue", "wed", "thu", "fri"]},
        "IsRepetitive": True,
        "activity_duration": None,
        "start_date": "2026-05-18",
        "IsArbitrary": False,
        "time": {"preferred": "07:00"},
        "notes": None,
    }


def _gym_specific_days() -> dict:
    return {
        "name": "gym",
        "frequency": {"type": "specific_days", "days": ["tue", "thu", "sat"]},
        "activity_duration": 120,
        "start_date": "2026-05-18",
        "time": {"preferred": "16:30"},
    }


def _watch_movies_daily() -> dict:
    return {
        "name": "watch movies",
        "frequency": {"type": "daily"},
        "activity_duration": 45,
        "start_date": "2026-05-18",
        "time": {"preferred": "22:00"},
    }


def _free_form_no_time() -> dict:
    return {
        "name": "long walk",
        "frequency": {"type": "weekly", "days": ["sat"]},
        "activity_duration": 120,
        "start_date": "2026-05-18",
        "time": {"preferred": "afternoon"},
    }


def test_specific_days_expands_to_each_weekday():
    payload = {"Activities": [_wake_up_workdays()]}
    out = expand_routines_to_assignments(
        payload,
        schedule_start=date(2026, 5, 18),  # Monday
        schedule_end=date(2026, 5, 24),  # Sunday
    )
    # 5 workdays in this week produce 5 wake-up rows.
    assert len(out) == 5
    assert [a["date"] for a in out] == ["2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22"]
    for a in out:
        assert a["start"] == "07:00"
        # IsInstantaneous gives a 5-min duration, so end is 07:05.
        assert a["end"] == "07:05"
        assert a["name"] == "wake up"
        assert "Z" not in a["start"]


def test_daily_expands_to_every_day():
    payload = {"Activities": [_watch_movies_daily()]}
    out = expand_routines_to_assignments(
        payload,
        schedule_start=date(2026, 5, 18),
        schedule_end=date(2026, 5, 24),
    )
    assert len(out) == 7
    assert out[0]["start"] == "22:00"
    assert out[0]["end"] == "22:45"


def test_time_of_day_alias_resolves_to_default_slot():
    """Aliases like `afternoon` resolve to a concrete slot so the routine lands on the calendar."""
    payload = {"Activities": [_free_form_no_time()]}
    out = expand_routines_to_assignments(
        payload,
        schedule_start=date(2026, 5, 18),
        schedule_end=date(2026, 5, 24),
    )
    # "long walk", weekly Sat, afternoon resolves to 14:00.
    assert len(out) == 1
    assert out[0]["date"] == "2026-05-23"
    assert out[0]["start"] == "14:00"
    assert out[0]["fallback_time"] is True


def test_specific_days_inside_one_week_window():
    payload = {"Activities": [_gym_specific_days()]}
    out = expand_routines_to_assignments(
        payload,
        schedule_start=date(2026, 5, 18),
        schedule_end=date(2026, 5, 24),
    )
    # Tue, Thu, Sat in the window produce 3 entries.
    assert [a["date"] for a in out] == ["2026-05-19", "2026-05-21", "2026-05-23"]


def test_string_input_is_accepted():
    import json

    payload = json.dumps({"Activities": [_wake_up_workdays()]})
    out = expand_routines_to_assignments(
        payload,
        schedule_start=date(2026, 5, 18),
        schedule_end=date(2026, 5, 22),
    )
    assert len(out) == 5


def test_merge_dedupes_same_slot():
    routine = [
        {"date": "2026-05-18", "start": "07:00", "end": "07:05", "name": "wake up"},
    ]
    scheduler = [
        # Same wake-up slot, expected to be deduped.
        {"date": "2026-05-18", "start": "07:00", "end": "07:05", "name": "wake up"},
        {"date": "2026-05-18", "start": "10:00", "end": "11:00", "name": "deep work"},
    ]
    merged = merge_assignments(routine, scheduler)
    assert len(merged) == 2
    names = sorted(a["name"] for a in merged)
    assert names == ["deep work", "wake up"]


def test_merge_routine_wins_for_same_key():
    routine = [
        {"date": "2026-05-18", "start": "07:00", "name": "wake up", "end": "07:05", "marker": "routine"},
    ]
    scheduler = [
        {"date": "2026-05-18", "start": "07:00", "name": "wake up", "end": "07:30", "marker": "scheduler"},
    ]
    merged = merge_assignments(routine, scheduler)
    assert len(merged) == 1
    assert merged[0]["marker"] == "routine"


def test_specific_days_ignores_bad_start_date():
    """When `frequency.days=[sat]`, a Wednesday `start_date` is ignored and Saturdays still appear."""
    payload = {
        "Activities": [
            {
                "name": "gym",
                "frequency": {"type": "specific_days", "days": ["sat"]},
                "activity_duration": 120,
                "start_date": "2026-05-20",
                "time": {"preferred": "16:30"},
            }
        ],
    }
    out = expand_routines_to_assignments(
        payload,
        schedule_start=date(2026, 5, 16),
        schedule_end=date(2026, 5, 21),
    )
    # The window contains one Saturday (2026-05-16); the bogus start_date must not suppress it.
    assert [a["date"] for a in out] == ["2026-05-16"]
    assert out[0]["start"] == "16:30"


def test_merge_drops_scheduler_routine_entries_by_name():
    """Scheduler entries matching an expander routine name are dropped regardless of date."""
    routine = [
        {"date": "2026-05-19", "start": "19:30", "end": "21:30", "name": "gym"},
        {"date": "2026-05-21", "start": "19:30", "end": "21:30", "name": "gym"},
    ]
    scheduler = [
        {"date": "2026-05-20", "start": "16:30", "end": "18:30", "name": "gym"},
        {"date": "2026-05-20", "start": "10:00", "end": "10:30", "name": "deep work"},
    ]
    merged = merge_assignments(routine, scheduler)
    names_and_dates = sorted((m["name"], m["date"]) for m in merged)
    assert ("gym", "2026-05-20") not in names_and_dates
    assert ("deep work", "2026-05-20") in names_and_dates


def test_null_time_preferred_uses_default_slot_when_days_are_known():
    """A null `time.preferred` with known days places the routine at the default morning slot."""
    payload = {
        "Activities": [
            {
                "name": "long walk",
                "frequency": {"type": "specific_days", "days": ["sat"]},
                "activity_duration": 120,
                "start_date": "2026-05-16",
                "time": {"preferred": None},
            }
        ],
    }
    out = expand_routines_to_assignments(
        payload,
        schedule_start=date(2026, 5, 16),  # Sat
        schedule_end=date(2026, 5, 22),
    )
    assert len(out) == 1
    assert out[0]["date"] == "2026-05-16"
    # Default morning slot with the activity's duration applied.
    assert out[0]["start"] == "10:00"
    assert out[0]["end"] == "12:00"
    assert out[0]["fallback_time"] is True


def test_null_time_and_no_days_is_still_deferred():
    """A routine with no days, no pattern, and no time is deferred to the LLM scheduler."""
    payload = {
        "Activities": [
            {
                "name": "something flexible",
                "frequency": {"type": None, "days": None},
                "activity_duration": 60,
                "start_date": "2026-05-16",
                "time": {"preferred": None},
            }
        ],
    }
    out = expand_routines_to_assignments(
        payload,
        schedule_start=date(2026, 5, 16),
        schedule_end=date(2026, 5, 22),
    )
    assert out == []


def test_duration_clamped_when_absurd():
    """A 1260-minute duration is clamped so the end time does not silently wrap past midnight."""
    payload = {
        "Activities": [
            {
                "name": "work",
                "frequency": {"type": "daily"},
                "activity_duration": 1260,
                "start_date": "2026-05-18",
                "time": {"preferred": "08:00"},
            }
        ],
    }
    out = expand_routines_to_assignments(
        payload,
        schedule_start=date(2026, 5, 18),
        schedule_end=date(2026, 5, 18),
    )
    assert len(out) == 1
    end = out[0]["end"]
    hh, mm = (int(p) for p in end.split(":"))
    # Clamped to 16h: 08:00 + 16h = 24:00, which is also 00:00 of the next day.
    assert (hh, mm) in {(0, 0), (24, 0)}, f"unexpected end: {end!r}"
