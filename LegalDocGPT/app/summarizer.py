from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.logger import log_info
import json

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["clause_data"],
    template="""
You are a legal assistant helping non-lawyers understand contracts.

Below is a structured analysis of a legal contract with extracted clauses and risk levels:

{clause_data}

Write a clear, plain-English summary for a non-expert. Structure your response as follows:

1. OVERVIEW: A 2-3 sentence plain English description of what this contract is about.
2. KEY OBLIGATIONS: What each party must do.
3. HIGH RISK AREAS: Flag any High risk clauses and explain why they matter in simple terms.
4. RECOMMENDATIONS: 2-3 practical suggestions for the person signing this contract.

Keep the language simple, avoid legal jargon, and be concise.
"""
)

def summarize_contract(clause_data: list) -> dict:
    """
    Takes structured clause data and generates a plain-English summary report.
    Returns a dict with clause_data and plain English summary.
    """
    log_info("Generating plain-English summary...")

    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    chain = SUMMARY_PROMPT | llm

    clause_text = json.dumps(clause_data, indent=2)
    response = chain.invoke({"clause_data": clause_text})
    summary_text = response.content

    high_risk = [c for c in clause_data if c.get("risk_level") == "High"]
    medium_risk = [c for c in clause_data if c.get("risk_level") == "Medium"]
    low_risk = [c for c in clause_data if c.get("risk_level") == "Low"]

    report = {
        "clauses": clause_data,
        "risk_summary": {
            "high": len(high_risk),
            "medium": len(medium_risk),
            "low": len(low_risk),
            "total": len(clause_data)
        },
        "plain_english_summary": summary_text
    }

    log_info("Summary report generated successfully.")
    return report
