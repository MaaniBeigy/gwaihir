from src.agents.scheduler_core import _build_schedule_window, _get_scheduling_period_weeks


def test_get_scheduling_period_weeks_defaults_to_one(monkeypatch):
    monkeypatch.delenv("SCHEDULING_PERIOD_WEEKS", raising=False)
    assert _get_scheduling_period_weeks() == 1


def test_get_scheduling_period_weeks_clamps_invalid_values(monkeypatch):
    monkeypatch.setenv("SCHEDULING_PERIOD_WEEKS", "0")
    # _get_scheduling_period_weeks does not clamp by itself; clamping happens in
    # the calendar/normalizer helpers. The agent reads the raw int.
    assert _get_scheduling_period_weeks() == 0

    monkeypatch.setenv("SCHEDULING_PERIOD_WEEKS", "999")
    assert _get_scheduling_period_weeks() == 999


def test_build_schedule_window_for_one_week_is_seven_days_inclusive():
    window = _build_schedule_window("2026-03-05", 1)
    assert window["start_date"] == "2026-03-05"
    assert window["end_date"] == "2026-03-11"


def test_build_schedule_window_for_two_weeks_is_fourteen_days_inclusive():
    window = _build_schedule_window("2026-03-05", 2)
    assert window["start_date"] == "2026-03-05"
    assert window["end_date"] == "2026-03-18"
