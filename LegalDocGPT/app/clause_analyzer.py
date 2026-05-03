import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from utils.logger import log_info, log_error

CLAUSE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
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
)

def analyze_clauses(contract_text: str) -> list:
    """
    Chunks contract text, embeds it into FAISS vector store,
    retrieves relevant sections, and extracts structured clause data.
    """
    log_info("Splitting contract into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(contract_text)
    log_info(f"Created {len(chunks)} chunks from contract.")

    log_info("Building FAISS vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": CLAUSE_PROMPT}
    )

    log_info("Analyzing clauses via RAG pipeline...")
    question = (
        "Extract all key legal clauses from this contract. "
        "Include clauses related to payment terms, termination, liability, "
        "confidentiality, intellectual property, indemnification, and dispute resolution."
    )

    result = qa_chain.invoke({"query": question})
    raw_output = result.get("result", "")

    try:
        clause_data = json.loads(raw_output)
        log_info(f"Successfully extracted {len(clause_data)} clauses.")
        return clause_data
    except json.JSONDecodeError:
        log_error("Failed to parse clause JSON. Returning raw output.")
        return [{"clause_name": "Raw Analysis", "description": raw_output,
                 "risk_level": "Unknown", "reason": "Could not parse structured output."}]
