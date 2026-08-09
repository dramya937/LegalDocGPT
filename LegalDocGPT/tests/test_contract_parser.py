import pytest
from app.contract_parser import parse_contract


def test_unsupported_file_type_raises_value_error(tmp_path):
    bad_file = tmp_path / "contract.txt"
    bad_file.write_text("This is not a supported format.")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_contract(str(bad_file))


def test_parses_docx(tmp_path):
    docx = pytest.importorskip("docx")
    file_path = tmp_path / "sample.docx"

    document = docx.Document()
    document.add_paragraph("This Agreement is entered into by Party A and Party B.")
    document.add_paragraph("Termination requires 30 days written notice.")
    document.save(str(file_path))

    text = parse_contract(str(file_path))

    assert "Party A" in text
    assert "Termination" in text
    # clean_text() should have collapsed the paragraph breaks into single spaces
    assert "\n" not in text
