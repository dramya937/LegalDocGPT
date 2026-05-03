# ⚖️ LegalDoc GPT — AI-Powered Contract Analysis

An AI-powered contract analysis tool that uses a **Retrieval-Augmented Generation (RAG)** pipeline to extract key clauses, assess risk levels, and generate plain-English summaries from legal documents — built for non-lawyers who need to understand contracts quickly.

---

## 🖥️ Demo

![LegalDoc GPT UI](assets/demo.png)

> Upload a PDF or DOCX contract → Get a structured clause breakdown with risk levels and a plain-English summary instantly.

---

## 🔧 How It Works

```
PDF / DOCX
    │
    ▼
Contract Parser (PyPDF2 / python-docx)
    │
    ▼
Text Chunker (LangChain RecursiveCharacterTextSplitter)
    │
    ▼
FAISS Vector Store (OpenAI Embeddings)
    │
    ▼
RAG Pipeline (RetrievalQA + GPT-4)
    │
    ▼
Structured JSON (Clauses + Risk Levels)
    │
    ▼
Plain-English Summary Report (Streamlit UI)
```

---

## ✨ Features

- **RAG-based clause extraction** — contracts are chunked and embedded so large documents don't overflow the context window
- **Structured risk scoring** — each clause is tagged as High / Medium / Low risk with a plain-English explanation
- **Plain-English summary** — non-lawyers can understand what they're signing
- **Streamlit UI** — clean browser-based interface with risk overview metrics and expandable clause cards
- **Download report** — export the full analysis as a JSON file
- **Supports PDF and DOCX** — handles both common contract formats

---

## 🗂️ Project Structure

```
LegalDocGPT/
│
├── app/
│   ├── contract_parser.py      # PDF/DOCX text extraction
│   ├── clause_analyzer.py      # LangChain RAG pipeline + FAISS
│   └── summarizer.py           # Plain-English report generation
│
├── utils/
│   ├── logger.py               # Logging utility
│   └── text_cleaner.py         # Text preprocessing
│
├── streamlit_app.py            # Streamlit web UI
├── main.py                     # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/dramya937/LegalDocGPT.git
cd LegalDocGPT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

You can set it via the Streamlit sidebar at runtime, or export it as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

### 4. Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

### 5. Or use the CLI
```bash
python main.py
```

---

## 📊 Sample Output

```json
{
  "clauses": [
    {
      "clause_name": "Termination Clause",
      "description": "Either party may terminate with 30 days written notice.",
      "risk_level": "Medium",
      "reason": "Short notice period may not allow adequate transition time."
    },
    {
      "clause_name": "Liability Cap",
      "description": "Vendor liability is capped at the total fees paid in the last 3 months.",
      "risk_level": "High",
      "reason": "Significantly limits compensation in case of major service failure."
    }
  ],
  "risk_summary": {
    "high": 2,
    "medium": 4,
    "low": 3,
    "total": 9
  },
  "plain_english_summary": "..."
}
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| LLM | GPT-4 (OpenAI) |
| RAG Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | OpenAI Embeddings |
| UI | Streamlit |
| Document Parsing | PyPDF2, python-docx |

---

## 💡 Why RAG Instead of Direct GPT?

Standard contract analysis tools send the entire document to GPT in one call. This breaks on large contracts (50+ pages) due to context window limits and produces lower quality results with no source grounding.

This tool uses RAG to:
- Split contracts into overlapping chunks (1000 tokens, 150 overlap)
- Embed and store chunks in a FAISS vector store
- Retrieve only the most relevant sections per query
- Ground GPT-4's responses in actual contract text

---

## 📌 Future Improvements

- [ ] Multi-contract comparison
- [ ] Clause-level highlighting in original document
- [ ] Support for scanned PDFs via OCR (Tesseract)
- [ ] Fine-tuned model on legal datasets

---

## 📄 License

MIT License
