from utils.text_cleaner import clean_text


def test_collapses_newlines_to_spaces():
    assert clean_text("Line one\nLine two") == "Line one Line two"


def test_collapses_multiple_whitespace():
    assert clean_text("Too    many     spaces") == "Too many spaces"


def test_strips_leading_and_trailing_whitespace():
    assert clean_text("  \n  padded text  \n  ") == "padded text"


def test_empty_string_stays_empty():
    assert clean_text("") == ""


def test_preserves_punctuation_and_content():
    text = "Section 4.2: Payment is due within 30 days."
    assert clean_text(text) == text
