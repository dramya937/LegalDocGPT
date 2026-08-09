"""
eval/run_ragas_eval.py
-----------------------
Evaluates the RAG pipeline (retrieval + generation) against a hand-labeled
ground-truth dataset (eval_dataset.json) using two complementary methods:

1. RAGAS metrics (needs an OpenAI API key — this is an LLM-judged eval):
   - Faithfulness:            does the answer stay grounded in retrieved context?
   - Answer Relevancy:        does the answer actually address the question?
   - Context Precision:       are the retrieved chunks relevant to the question?
   - Context Recall:          did retrieval surface what the ground truth needs?

2. Clause coverage (free, deterministic, no LLM judge needed):
   For each test contract, checks whether the full analyze_clauses() pipeline
   surfaced every clause type we know is in the contract (payment terms,
   termination, liability, confidentiality, etc). This is a cheap sanity
   check you can run on every change without spending API credits, and it's
   what actually catches "the model silently dropped a clause type."

Usage:
    export OPENAI_API_KEY=your_key_here
    python eval/run_ragas_eval.py

Cost note: this makes a small number of real OpenAI API calls (embeddings +
gpt-4o-mini by default for eval questions, gpt-4o for the full clause
extraction pass). Expect well under $0.50 total for the 3 test contracts.
Results (including realized cost) are written to eval/results/.
"""

import sys
import os
import json
import types
from pathlib import Path

# --- Workaround for a real upstream ragas bug (as of ragas 0.2.x-0.4.3) ---
# ragas.llms.base unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai, a module that no longer exists
# in langchain-community >= 0.3.something (Google Vertex AI integration
# moved to its own standalone package). We never use Vertex AI — this
# project runs entirely on OpenAI — so we stub the dead import path instead
# of downgrading the whole LangChain stack to satisfy an import path we
# never call. Remove this once ragas fixes the upstream import.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
class _ChatVertexAIStub:
    pass
_stub.ChatVertexAI = _ChatVertexAIStub
sys.modules["langchain_community.chat_models.vertexai"] = _stub
# --- end workaround ---

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from app.clause_analyzer import build_vectorstore, analyze_clauses
from utils.text_cleaner import clean_text
from utils.cost_tracker import CostTracker, DEFAULT_MODEL

EVAL_DIR = Path(__file__).resolve().parent
CONTRACTS_DIR = EVAL_DIR / "test_contracts"
RESULTS_DIR = EVAL_DIR / "results"

EVAL_QUESTION_MODEL = "gpt-4o-mini"  # cheap model is fine for answering single eval questions

# Ground-truth clause types we know exist in each synthetic test contract,
# used for the free clause-coverage check.
EXPECTED_CLAUSE_TYPES = {
    "contract_1_service_agreement.txt": [
        "payment", "termination", "liability", "confidentiality",
        "intellectual property", "indemnif", "dispute",
    ],
    "contract_2_nda.txt": [
        "confidential", "term", "exclu", "remed",
    ],
    "contract_3_vendor_lease.txt": [
        "payment", "termination", "liability", "indemnif", "insurance", "renewal",
    ],
}


def answer_question(question: str, contexts: list[str], model: str = EVAL_QUESTION_MODEL) -> str:
    """Simple single-turn QA over retrieved context, used only for the RAGAS
    eval (separate from the full clause-extraction pipeline, which answers a
    fixed extraction prompt rather than arbitrary eval questions)."""
    context_str = "\n\n".join(contexts)
    prompt = (
        f"Answer the question using ONLY the contract excerpts below. "
        f"Be concise and specific.\n\nContract Context:\n{context_str}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke(prompt)
    return response.content


def run_ragas_eval(tracker: CostTracker) -> dict:
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings

    eval_items = json.loads((EVAL_DIR / "eval_dataset.json").read_text())

    # Group by contract so we only embed each contract once
    by_contract = {}
    for item in eval_items:
        by_contract.setdefault(item["contract_file"], []).append(item)

    rows = []
    for contract_file, items in by_contract.items():
        raw_text = (CONTRACTS_DIR / contract_file).read_text()
        contract_text = clean_text(raw_text)
        vectorstore, _ = build_vectorstore(contract_text)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        for item in items:
            question = item["question"]
            with tracker.track(f"eval_retrieval::{contract_file}", model=EVAL_QUESTION_MODEL) as t:
                retrieved_docs = retriever.invoke(question)
                t.record_manual(input_tokens=0, output_tokens=0)  # embeddings priced separately by OpenAI
            contexts = [doc.page_content for doc in retrieved_docs]

            with tracker.track(f"eval_answer::{contract_file}", model=EVAL_QUESTION_MODEL) as t:
                answer = answer_question(question, contexts)
                # rough estimate since we're not capturing the raw response object here
                t.record_manual(
                    input_tokens=tracker.records[-1].input_tokens if tracker.records else 0,
                    output_tokens=0,
                )

            rows.append({
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": item["ground_truth"],
            })

    dataset = EvaluationDataset.from_list(rows)

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    return result.to_pandas().to_dict(orient="records")


def run_clause_coverage_check(tracker: CostTracker) -> dict:
    """Runs the real, full analyze_clauses() pipeline (same code path the
    app uses) against each test contract and checks whether every known
    clause type got surfaced somewhere in the output. No LLM judge, no
    ground-truth 'correct answer' needed — just: did it show up at all."""
    coverage_results = []

    for contract_file, expected_types in EXPECTED_CLAUSE_TYPES.items():
        raw_text = (CONTRACTS_DIR / contract_file).read_text()
        contract_text = clean_text(raw_text)

        analysis = analyze_clauses(contract_text, model=DEFAULT_MODEL, tracker=tracker)
        clauses = analysis["clauses"]

        combined_text = " ".join(
            f"{c.get('clause_name', '')} {c.get('description', '')}".lower()
            for c in clauses
        )

        found = [t for t in expected_types if t in combined_text]
        missing = [t for t in expected_types if t not in combined_text]

        coverage_results.append({
            "contract_file": contract_file,
            "expected_clause_types": expected_types,
            "found": found,
            "missing": missing,
            "coverage_pct": round(100 * len(found) / len(expected_types), 1),
            "clauses_extracted": len(clauses),
        })

    return coverage_results


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY before running this eval.")
        print("This eval makes real (small, ~$0.10-0.50 total) OpenAI API calls.")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)
    tracker = CostTracker()

    print("=" * 60)
    print("Running clause coverage check (deterministic, no LLM judge)...")
    print("=" * 60)
    coverage = run_clause_coverage_check(tracker)
    for r in coverage:
        print(f"\n{r['contract_file']}: {r['coverage_pct']}% coverage "
              f"({len(r['found'])}/{len(r['expected_clause_types'])} clause types found)")
        if r["missing"]:
            print(f"  MISSING: {r['missing']}")

    (RESULTS_DIR / "clause_coverage.json").write_text(json.dumps(coverage, indent=2))

    print("\n" + "=" * 60)
    print("Running RAGAS evaluation (faithfulness, relevancy, precision, recall)...")
    print("=" * 60)
    try:
        ragas_scores = run_ragas_eval(tracker)
        (RESULTS_DIR / "ragas_scores.json").write_text(json.dumps(ragas_scores, indent=2, default=str))
        print(f"\nSaved per-question RAGAS scores to {RESULTS_DIR / 'ragas_scores.json'}")

        # Print averaged scores across all eval questions
        if ragas_scores:
            metric_keys = [k for k in ragas_scores[0].keys()
                            if k not in ("user_input", "retrieved_contexts", "response", "reference")]
            print("\nAverage scores across all eval questions:")
            for key in metric_keys:
                vals = [r[key] for r in ragas_scores if isinstance(r.get(key), (int, float))]
                if vals:
                    print(f"  {key}: {sum(vals)/len(vals):.3f}")
    except Exception as e:
        print(f"\nRAGAS evaluation failed: {e}")
        print("Clause coverage results above are still valid and were saved.")

    totals = tracker.summary()
    print("\n" + "=" * 60)
    print(f"Total eval run cost: ${totals['total_cost_usd']:.4f}")
    print(f"Total eval run latency: {totals['total_latency_seconds']:.1f}s")
    print("=" * 60)
    (RESULTS_DIR / "run_cost_summary.json").write_text(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
