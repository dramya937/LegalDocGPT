import os
import json
import time
import concurrent.futures
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

CATEGORY_PROMPT_TEMPLATE = """
You are a legal contract analysis expert. Using the contract excerpts below,
extract and analyze any clauses related to: {category}

Contract Context:
{context}

Respond ONLY with a valid JSON array in this exact format:
[
  {{
    "clause_name": "Clause Title",
    "description": "What this clause says in simple terms",
    "risk_level": "High | Medium | Low",
    "reason": "Why this risk level was assigned"
  }}
]

If the contract excerpts above do not contain any clause related to this
category, respond with an empty array: []
Do not include any text outside the JSON array.
"""

# One retrieval query *and* one dedicated extraction call per clause
# category, rather than one merged call for everything.
#
# Merging all categories' retrieved chunks into a single extraction prompt
# (the previous approach) still let the model silently fold or drop entire
# categories — e.g. Non-Compete or Warranties could vanish even when their
# source text was present in the context, because the model just didn't
# choose to emit them alongside everything else it was asked to find in one
# pass. Giving each category its own call, its own retrieved context, and
# an explicit "empty array if nothing found" instruction removes that
# failure mode: a category can only be correctly-empty, never silently
# skipped.
CLAUSE_CATEGORIES = [
    "payment terms, fees, and invoicing",
    "termination and term of the agreement",
    "limitation of liability and damages",
    "confidentiality and non-disclosure obligations",
    "intellectual property ownership and licensing",
    "indemnification obligations",
    "warranties, disclaimers, and representations",
    "dispute resolution, governing law, and venue",
    "non-solicitation and non-compete restrictions",
]


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

    Strips markdown code fences first, since models sometimes wrap JSON in
```json ... ``` blocks despite being told not to.
    """
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned.lstrip("`")
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()

    try:
        clause_data = json.loads(cleaned)
        log_info(f"Successfully extracted {len(clause_data)} clauses.")
        return clause_data
    except json.JSONDecodeError:
        log_error("Failed to parse clause JSON. Returning raw output.")
        return [{"clause_name": "Raw Analysis", "description": raw_output,
                 "risk_level": "Unknown", "reason": "Could not parse structured output."}]


def _extract_for_category(category: str, context_str: str, model: str) -> dict:
    """Runs one category's extraction call. Executed in a worker thread by
    analyze_clauses(), so this must not touch shared mutable state (like a
    CostTracker) directly — it returns raw numbers instead, and the caller
    records them on the main thread once all futures complete."""
    prompt = CATEGORY_PROMPT_TEMPLATE.format(category=category, context=context_str)
    llm = ChatOpenAI(model=model, temperature=0)

    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    clauses = parse_clause_json(response.content)
    # Drop the "no match" case cleanly rather than keeping an empty-array
    # no-op entry in the merged results.
    if isinstance(clauses, list) and len(clauses) == 1 and not clauses[0].get("clause_name"):
        clauses = []

    return {
        "category": category,
        "clauses": clauses,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "latency_seconds": elapsed,
    }


def analyze_clauses(
    contract_text: str,
    model: str = DEFAULT_MODEL,
    k: int = 8,
    tracker: CostTracker = None,
    max_workers: int = 6,
) -> dict:
    """
    Chunks contract text, embeds it into a FAISS vector store, then for each
    clause category (see CLAUSE_CATEGORIES): retrieves that category's most
    relevant chunks and runs a dedicated extraction call. Calls run
    concurrently (up to max_workers at a time) so per-category extraction
    doesn't multiply wall-clock latency by the number of categories.

    Returns a dict with:
      - "clauses": merged list of extracted clause dicts across all categories
      - "retrieved_contexts": the union of chunk texts retrieved across all
        categories (needed for RAGAS context_precision / context_recall /
        faithfulness scoring)
      - "cost_usd" / "latency_seconds": this stage's total cost and latency
        (latency is wall-clock across the concurrent calls, not the sum of
        each call's individual latency)
    """
    tracker = tracker or CostTracker()

    vectorstore, all_chunks = build_vectorstore(contract_text)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(k * 4, len(all_chunks))},
    )

    log_info(f"Retrieving top-{k} chunks for each of {len(CLAUSE_CATEGORIES)} clause categories...")
    category_contexts = {}
    all_retrieved = set()
    for category in CLAUSE_CATEGORIES:
        docs = retriever.invoke(category)
        chunks_for_category = [doc.page_content for doc in docs]
        category_contexts[category] = "\n\n".join(chunks_for_category)
        all_retrieved.update(chunks_for_category)
    log_info(f"Retrieved {len(all_retrieved)} unique chunks (of {len(all_chunks)} total) across all categories.")

    log_info(f"Running {len(CLAUSE_CATEGORIES)} per-category extraction calls (up to {max_workers} concurrent)...")
    wall_clock_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_for_category, category, category_contexts[category], model): category
            for category in CLAUSE_CATEGORIES
        }
        for future in concurrent.futures.as_completed(futures):
            category = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                log_error(f"Extraction failed for category '{category}': {e}")
    wall_clock_elapsed = time.perf_counter() - wall_clock_start

    # Record bookkeeping on the main thread — CostTracker isn't designed to
    # be written to from multiple threads at once, so we do this after every
    # future has already completed, rather than inside _extract_for_category.
    total_input_tokens = 0
    total_output_tokens = 0
    merged_clauses = []
    seen_names = set()
    for result in sorted(results, key=lambda r: r["category"]):
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        for clause in result["clauses"]:
            name_key = (clause.get("clause_name") or "").strip().lower()
            if name_key and name_key not in seen_names:
                seen_names.add(name_key)
                merged_clauses.append(clause)

    with tracker.track("clause_extraction", model=model) as t:
        t.record_manual(total_input_tokens, total_output_tokens)
    # The tracker's own timer would only measure this bookkeeping block, not
    # the actual (concurrent) API calls above, so overwrite it with the real
    # wall-clock time the extraction calls took.
    tracker.records[-1].latency_seconds = round(wall_clock_elapsed, 2)

    stage_summary = tracker.summary()["by_stage"][-1]
    return {
        "clauses": merged_clauses,
        "retrieved_contexts": list(all_retrieved),
        "cost_usd": stage_summary["cost_usd"],
        "latency_seconds": stage_summary["latency_seconds"],
    }
