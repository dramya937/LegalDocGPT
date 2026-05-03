from app.contract_parser import parse_contract
from app.clause_analyzer import analyze_clauses
from app.summarizer import summarize_contract
from utils.logger import log_info

def main():
    file_path = input("Enter the contract file path (PDF/DOCX): ")
    log_info(f"Processing file: {file_path}")

    # Step 1: Parse contract
    contract_text = parse_contract(file_path)

    # Step 2: Analyze clauses
    clause_data = analyze_clauses(contract_text)

    # Step 3: Summarize contract
    summary_report = summarize_contract(clause_data)

    # Step 4: Output results
    print("\n=== Summary Report ===")
    print(summary_report)

if __name__ == "__main__":
    main()
