import os
import json
try:
    # langchain >= 0.3 moved text splitters into their own package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # fallback for older langchain (<0.3) where it still lived in core
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils.logger import log_info, log_error
from utils.cost_tracker import CostTracker, DEFAULT_MODEL

CLAUSE_PROMPT_TEMPLATE = """
You are a legal contract analysis expert. Using the contract excerpts below, extract and analyze key clauses.

Contract Context:
{context}

Task: {question}

Respond ONLY with a valid JSON array in this exact format:
[
  {{
    "clause_name": "Clause Title",
    "description": "What this clause says in simple terms",
    "risk_level": "High | Medium | Low",
    "reason": "Why this risk level was assigned"
  }}
]

Do not include any text outside the JSON array.
"""

CLAUSE_QUESTION = (
    "Extract all key legal clauses from this contract. "
    "Include clauses related to payment terms, termination, liability, "
    "confidentiality, intellectual property, indemnification, and dispute resolution."
)


def build_vectorstore(contract_text: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Chunk contract text and embed it into a FAISS vector store. Split out
    from analyze_clauses() so the eval script can reuse it directly to test
    retrieval quality without also paying for a generation call."""
    log_info("Splitting contract into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(contract_text)
    log_info(f"Created {len(chunks)} chunks from contract.")

    log_info("Building FAISS vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore, chunks


def parse_clause_json(raw_output: str) -> list:
    """
    Parses the LLM's raw output into structured clause data, falling back
    to a single 'Raw Analysis' entry if the model didn't return valid JSON.
    Pulled out as its own function so it can be unit tested with canned
    LLM outputs (including malformed ones) without making an API call.
    """
    try:
        clause_data = json.loads(raw_output)
        log_info(f"Successfully extracted {len(clause_data)} clauses.")
        return clause_data
    except json.JSONDecodeError:
        log_error("Failed to parse clause JSON. Returning raw output.")
        return [{"clause_name": "Raw Analysis", "description": raw_output,
                 "risk_level": "Unknown", "reason": "Could not parse structured output."}]


def analyze_clauses(
    contract_text: str,
    model: str = DEFAULT_MODEL,
    k: int = 5,
    tracker: CostTracker = None,
) -> dict:
    """
    Chunks contract text, embeds it into a FAISS vector store, retrieves
    the k most relevant chunks, and extracts structured clause data.

    Returns a dict with:
      - "clauses": list of extracted clause dicts
      - "retrieved_contexts": the chunk texts actually retrieved (needed
        for RAGAS context_precision / context_recall / faithfulness scoring)
      - "cost_usd" / "latency_seconds": this stage's cost and latency
    """
    tracker = tracker or CostTracker()

    vectorstore, _ = build_vectorstore(contract_text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    log_info(f"Retrieving top-{k} relevant chunks...")
    retrieved_docs = retriever.invoke(CLAUSE_QUESTION)
    retrieved_contexts = [doc.page_content for doc in retrieved_docs]
    context_str = "\n\n".join(retrieved_contexts)

    prompt = CLAUSE_PROMPT_TEMPLATE.format(context=context_str, question=CLAUSE_QUESTION)
    llm = ChatOpenAI(model=model, temperature=0)

    log_info("Analyzing clauses via RAG pipeline...")
    with tracker.track("clause_extraction", model=model) as t:
        response = llm.invoke(prompt)
        t.record(response)
    raw_output = response.content
    clause_data = parse_clause_json(raw_output)

    stage_summary = tracker.summary()["by_stage"][-1]
    return {
        "clauses": clause_data,
        "retrieved_contexts": retrieved_contexts,
        "cost_usd": stage_summary["cost_usd"],
        "latency_seconds": stage_summary["latency_seconds"],
    }
