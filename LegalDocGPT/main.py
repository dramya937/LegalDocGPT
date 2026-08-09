from app.contract_parser import parse_contract
from app.clause_analyzer import analyze_clauses
from app.summarizer import summarize_contract
from utils.logger import log_info
from utils.cost_tracker import CostTracker, DEFAULT_MODEL

def main():
    file_path = input("Enter the contract file path (PDF/DOCX): ")
    log_info(f"Processing file: {file_path}")

    tracker = CostTracker()

    # Step 1: Parse contract
    contract_text = parse_contract(file_path)

    # Step 2: Analyze clauses (RAG: retrieve + extract)
    analysis = analyze_clauses(contract_text, model=DEFAULT_MODEL, tracker=tracker)

    # Step 3: Summarize contract
    summary_report = summarize_contract(analysis["clauses"], model=DEFAULT_MODEL, tracker=tracker)

    # Step 4: Output results
    print("\n=== Summary Report ===")
    print(summary_report["plain_english_summary"])

    totals = tracker.summary()
    print("\n=== Cost & Latency ===")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Total cost: ${totals['total_cost_usd']:.5f}")
    print(f"Total latency: {totals['total_latency_seconds']:.2f}s")
    print(f"Tokens: {totals['total_input_tokens']} in / {totals['total_output_tokens']} out")

if __name__ == "__main__":
    main()
