from app.clause_analyzer import parse_clause_json


VALID_LLM_OUTPUT = """[
  {
    "clause_name": "Termination Clause",
    "description": "Either party may terminate with 30 days notice.",
    "risk_level": "Medium",
    "reason": "Short notice period."
  }
]"""

MALFORMED_LLM_OUTPUT = "Sure! Here are the clauses: [not valid json..."

EMPTY_ARRAY_OUTPUT = "[]"


def test_parses_valid_json_array():
    result = parse_clause_json(VALID_LLM_OUTPUT)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["clause_name"] == "Termination Clause"
    assert result[0]["risk_level"] == "Medium"


def test_falls_back_gracefully_on_malformed_json():
    """This is the failure mode that matters most in production: the model
    doesn't always obey 'respond ONLY with JSON'. The pipeline should
    degrade to a labeled raw-text entry, not crash."""
    result = parse_clause_json(MALFORMED_LLM_OUTPUT)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["risk_level"] == "Unknown"
    assert result[0]["clause_name"] == "Raw Analysis"
    assert MALFORMED_LLM_OUTPUT in result[0]["description"]


def test_handles_empty_array():
    result = parse_clause_json(EMPTY_ARRAY_OUTPUT)
    assert result == []
