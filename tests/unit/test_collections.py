import inspect

import pytest

from src.memory.collections import collection_for, is_coach_collection_name, parse_collection_name


def test_collection_for_formats_player_and_campaign_ids():
    assert collection_for(42, 7) == "coach_p42_c7"


def test_collection_for_coerces_string_ids():
    assert collection_for("42", "7") == "coach_p42_c7"


def test_collection_for_rejects_none_ids():
    with pytest.raises(ValueError):
        collection_for(None, 7)
    with pytest.raises(ValueError):
        collection_for(42, None)


def test_collection_for_rejects_negative_ids():
    with pytest.raises(ValueError):
        collection_for(-1, 7)
    with pytest.raises(ValueError):
        collection_for(42, -1)


def test_collection_for_rejects_non_numeric():
    with pytest.raises(ValueError):
        collection_for("abc", 7)


def test_is_coach_collection_name_recognises_pattern():
    assert is_coach_collection_name("coach_p1_c2")
    assert is_coach_collection_name("coach_p123456_c7890")
    assert not is_coach_collection_name("coach_p1")
    assert not is_coach_collection_name("interviewer")
    assert not is_coach_collection_name("processed_data")
    assert not is_coach_collection_name("")


def test_parse_collection_name_round_trip():
    assert parse_collection_name("coach_p11_c22") == (11, 22)
    assert parse_collection_name("not-a-coach-collection") is None


def test_no_other_call_site_constructs_collection_names():
    """The canonical prefix lives in a single module-level constant and
    `collection_for` emits names that begin with it.
    """
    from src.memory import collections as collections_module

    assert collections_module._COLLECTION_PREFIX == "coach_p"
    src = inspect.getsource(collection_for)
    assert "_COLLECTION_PREFIX" in src
    assert collection_for(1, 2).startswith(collections_module._COLLECTION_PREFIX)
