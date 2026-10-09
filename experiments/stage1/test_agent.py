import pytest

from experiments.stage1.agent import AgentResponseError, _parse_decision


def test_parse_valid_decision():
    action, expectation = _parse_decision(
        '{"action":"SEARCH","expectation":"Reveal adjacent movement costs."}'
    )
    assert action == "SEARCH"
    assert expectation == "Reveal adjacent movement costs."


def test_parse_fenced_json():
    action, _ = _parse_decision('\\x60\\x60\\x60json\\n{"action":"MOVE_E","expectation":"Move east."}\\n\\x60\\x60\\x60')
    assert action == "MOVE_E"


@pytest.mark.parametrize("content", ["not json", "[]", '{"expectation":"move"}', '{"action":"FLY"}'])
def test_reject_invalid_decisions(content):
    with pytest.raises(AgentResponseError):
        _parse_decision(content)
