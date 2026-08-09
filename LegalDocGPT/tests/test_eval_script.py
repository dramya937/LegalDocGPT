import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import run_ragas_eval as eval_mod


def test_coverage_detects_present_and_missing_clause_types():
    expected_types = ["payment", "termination", "liability"]
    fake_clauses = [
        {"clause_name": "Payment Terms", "description": "Invoices due in 30 days."},
        {"clause_name": "Termination", "description": "Either party may terminate with notice."},
        # liability deliberately absent
    ]
    combined_text = " ".join(
        f"{c.get('clause_name', '')} {c.get('description', '')}".lower() for c in fake_clauses
    )
    found = [t for t in expected_types if t in combined_text]
    missing = [t for t in expected_types if t not in combined_text]

    assert found == ["payment", "termination"]
    assert missing == ["liability"]


def test_expected_clause_types_defined_for_every_test_contract():
    contracts_dir = Path(__file__).resolve().parent.parent / "eval" / "test_contracts"
    contract_files = {p.name for p in contracts_dir.glob("*.txt")}
    expected_keys = set(eval_mod.EXPECTED_CLAUSE_TYPES.keys())
    assert contract_files == expected_keys, (
        f"Mismatch between test_contracts/ files and EXPECTED_CLAUSE_TYPES keys: "
        f"{contract_files.symmetric_difference(expected_keys)}"
    )


def test_eval_dataset_references_only_existing_contracts():
    import json
    eval_dir = Path(__file__).resolve().parent.parent / "eval"
    dataset = json.loads((eval_dir / "eval_dataset.json").read_text())
    contract_files = {p.name for p in (eval_dir / "test_contracts").glob("*.txt")}

    for item in dataset:
        assert item["contract_file"] in contract_files
        assert item["question"].strip() != ""
        assert item["ground_truth"].strip() != ""
