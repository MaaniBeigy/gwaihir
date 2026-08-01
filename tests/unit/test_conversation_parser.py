from src.agents.tools.conversation_parser import parse_conversation_history


def test_parse_string_strips_blank_lines_and_normalizes_escaped_newlines():
    raw = "Q: When do you wake up?\\nA: 07:00\n\n  \nQ: Lunch?\nA: noon\n"
    cleaned = parse_conversation_history(raw)
    assert cleaned == ("Q: When do you wake up?\nA: 07:00\nQ: Lunch?\nA: noon")


def test_parse_list_concatenates_entries():
    raw = ["Q: a\\nA: 1", "Q: b\nA: 2"]
    cleaned = parse_conversation_history(raw)
    assert cleaned == "Q: a\nA: 1\nQ: b\nA: 2"


def test_parse_empty_string():
    assert parse_conversation_history("") == ""


def test_parse_empty_list():
    assert parse_conversation_history([]) == ""


def test_parse_list_with_empty_entries():
    cleaned = parse_conversation_history(["", "  \n", "Q: real\nA: yes"])
    assert cleaned == "Q: real\nA: yes"
