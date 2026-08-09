import streamlit as st
import os
import tempfile
from app.contract_parser import parse_contract
from app.clause_analyzer import analyze_clauses
from app.summarizer import summarize_contract
from utils.cost_tracker import CostTracker, MODEL_PRICING, DEFAULT_MODEL

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LegalDoc GPT",
    page_icon="⚖️",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚖️ LegalDoc GPT")
st.markdown("**AI-powered contract analysis. Upload a contract and get plain-English insights instantly.**")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API key set!")
    st.divider()
    model = st.selectbox(
        "Model",
        options=list(MODEL_PRICING.keys()),
        index=list(MODEL_PRICING.keys()).index(DEFAULT_MODEL),
        help="Cheaper models cost less but may extract clauses less reliably — see the Evaluation section in the README for measured tradeoffs.",
    )
    pricing = MODEL_PRICING[model]
    st.caption(f"${pricing['input']:.2f}/1M input · ${pricing['output']:.2f}/1M output tokens")
    st.divider()
    st.markdown("**Supported formats:** PDF, DOCX")
    st.markdown("**Pipeline:** RAG-based clause extraction")

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📄 Upload your contract (PDF or DOCX)",
    type=["pdf", "docx"]
)

if uploaded_file:
    st.success(f"Uploaded: **{uploaded_file.name}**")

    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
    else:
        if st.button("🔍 Analyze Contract", type="primary", use_container_width=True):
            with st.spinner("Parsing contract..."):
                # Save uploaded file to temp location
                suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                contract_text = parse_contract(tmp_path)
                os.unlink(tmp_path)

            tracker = CostTracker()

            with st.spinner("Running RAG pipeline — extracting clauses..."):
                analysis = analyze_clauses(contract_text, model=model, tracker=tracker)

            with st.spinner("Generating plain-English summary..."):
                report = summarize_contract(analysis["clauses"], model=model, tracker=tracker)

            st.success("Analysis complete!")
            st.divider()

            # ── Risk Summary Cards ────────────────────────────────────────────
            st.subheader("📊 Risk Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Clauses", report["risk_summary"]["total"])
            col2.metric("🔴 High Risk", report["risk_summary"]["high"])
            col3.metric("🟡 Medium Risk", report["risk_summary"]["medium"])
            col4.metric("🟢 Low Risk", report["risk_summary"]["low"])
            st.divider()

            # ── Cost & Latency ───────────────────────────────────────────────
            totals = tracker.summary()
            st.subheader("💵 Cost & Latency (this run)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Model", model)
            c2.metric("Cost", f"${totals['total_cost_usd']:.5f}")
            c3.metric("Latency", f"{totals['total_latency_seconds']:.2f}s")
            with st.expander("Breakdown by pipeline stage"):
                st.table(totals["by_stage"])
            st.divider()

            # ── Plain English Summary ─────────────────────────────────────────
            st.subheader("📝 Plain English Summary")
            st.markdown(report["plain_english_summary"])
            st.divider()

            # ── Clause Breakdown ──────────────────────────────────────────────
            st.subheader("📋 Clause-by-Clause Breakdown")

            risk_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Unknown": "⚪"}

            for clause in report["clauses"]:
                risk = clause.get("risk_level", "Unknown")
                icon = risk_colors.get(risk, "⚪")

                with st.expander(f"{icon} {clause.get('clause_name', 'Unnamed Clause')} — {risk} Risk"):
                    st.markdown(f"**What it says:** {clause.get('description', 'N/A')}")
                    st.markdown(f"**Why this risk level:** {clause.get('reason', 'N/A')}")

            st.divider()

            # ── Download Report ───────────────────────────────────────────────
            import json
            report_json = json.dumps(report, indent=2)
            st.download_button(
                label="⬇️ Download Full Report (JSON)",
                data=report_json,
                file_name="legaldoc_report.json",
                mime="application/json"
            )

else:
    st.info("👆 Upload a PDF or DOCX contract to get started.")

    # ── How it works ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔧 How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.markdown("Upload any PDF or DOCX contract. The tool parses and cleans the text automatically.")
    with col2:
        st.markdown("### 2️⃣ RAG Analysis")
        st.markdown("The contract is chunked, embedded into a FAISS vector store, and analyzed using GPT-4 via a retrieval-augmented pipeline.")
    with col3:
        st.markdown("### 3️⃣ Report")
        st.markdown("Get a structured breakdown of all key clauses with risk levels and a plain-English summary you can actually understand.")
