"""
streamlit_app.py
Interactive Streamlit UI for Theme 2B: Open-Source Contributor Guide.
Runs natively with local Python environment.
"""

import os
import glob
import streamlit as st
from rag_engine import ContributorGuideRAG, REFUSAL_MESSAGE

st.set_page_config(
    page_title="Open-Source Contributor Guide (Gemma 4 RAG)",
    page_icon="🛠️",
    layout="wide"
)

# Custom Styling for polished aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .citation-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 12px;
        border-radius: 4px;
        margin-top: 10px;
    }
    .refusal-badge {
        color: #DC2626;
        font-weight: 600;
        background-color: #FEE2E2;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .success-badge {
        color: #059669;
        font-weight: 600;
        background-color: #D1FAE5;
        padding: 4px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_engine():
    engine = ContributorGuideRAG(chunk_size=120, chunk_overlap=30, top_k=3, similarity_threshold=0.28)
    files = glob.glob("docs/*.md")
    if files:
        engine.index_files(files)
    return engine

rag = get_rag_engine()

# Header
st.markdown("<div class='main-header'>🛠️ Open-Source Contributor Guide</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Powered by Google Gemma 4 RAG • Grounded Answers with Citations & Refusal Guardrails</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration & Docs")
    st.info(f"**Loaded Documents**:\n" + "\n".join([f"- `{d}`" for d in rag.knowledge_base["docs"]]))
    
    st.markdown("---")
    st.subheader("RAG Parameters")
    st.write(f"- **Chunk Size**: {rag.chunk_size} words")
    st.write(f"- **Chunk Overlap**: {rag.chunk_overlap} words")
    st.write(f"- **Top-K Retrieval**: {rag.top_k}")
    st.write(f"- **Relevance Threshold**: {rag.similarity_threshold}")
    st.write(f"- **Embedding Mode**: `{rag.embedder.mode}`")
    
    if st.button("🔄 Reload Default Docs"):
        rag.index_files(glob.glob("docs/*.md"))
        st.success("Docs re-indexed!")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Ask Assistant",
    "🚀 Where Do I Start?",
    "🔬 RAG vs No-RAG",
    "📊 Evaluation Report"
])

with tab1:
    st.subheader("Ask Questions About Contributing")
    
    # Quick question selector
    preset_questions = [
        "How do I set up the project on my computer?",
        "What command should I run to execute the test suite?",
        "What is the process for submitting a pull request?",
        "Which coding style rules must I follow for formatting?",
        "Where can I report violations of the Code of Conduct?",
        "Can I contribute without writing Python code?",
        "Who won the 2022 FIFA World Cup in Qatar?",
        "What is the minimum attendance percentage required for exams?"
    ]
    
    selected_example = st.selectbox("💡 Choose an example question or type below:", ["-- Custom question --"] + preset_questions)
    
    if selected_example != "-- Custom question --":
        default_query = selected_example
    else:
        default_query = ""
        
    user_query = st.text_input("Enter your question:", value=default_query, placeholder="e.g. How do I run tests with coverage?")
    
    if st.button("Submit Question", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant passages and synthesizing answer..."):
                answer_text, passages, was_refused = rag.answer(user_query)
                
            if was_refused:
                st.markdown("<span class='refusal-badge'>⚠️ Request Declined (Out of Scope / Low Relevance)</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='success-badge'>✅ Grounded Answer (Citing Official Docs)</span>", unsafe_allow_html=True)
                
            st.markdown(f"### Answer\n{answer_text}")
            
            with st.expander("🔎 Retrieved Documentation Passages (Full Transparency)", expanded=False):
                if not passages:
                    st.write("No passages retrieved.")
                for idx, p in enumerate(passages, start=1):
                    st.markdown(f"**Passage {idx}** — Document: `{p['doc']}` | Section: *{p['section']}* | Similarity Score: **{p['score']:.2f}**")
                    st.info(p["text"])

with tab2:
    st.subheader("🚀 'Where Do I Start?' Onboarding Checklist")
    st.write(
        "New contributors often face friction piecing together instructions from multiple documentation files. "
        "Click the button below to synthesize a verified, step-by-step onboarding plan from the repository's setup, "
        "contributing, and style guides."
    )
    if st.button("Generate Onboarding Checklist", type="primary"):
        with st.spinner("Compiling onboarding checklist from docs..."):
            checklist, _ = rag.generate_onboarding_checklist()
            st.markdown(checklist)

with tab3:
    st.subheader("🔬 RAG vs. No-RAG Comparison")
    st.write(
        "Compare an answer produced **with RAG** (grounded in the specific repository's docs) "
        "versus **without RAG** (unassisted LLM relying on general training data)."
    )
    comp_q = st.text_input("Comparison Question:", value="Which linter and code formatting tool does this project use?")
    if st.button("Compare Side-by-Side"):
        with st.spinner("Generating comparison..."):
            res = rag.compare_rag_vs_norag(comp_q)
            col1, col2 = st.columns(2)
            with col1:
                st.success("### ✅ With RAG (Doc-Grounded)")
                st.write(res["rag_answer"])
            with col2:
                st.warning("### ⚠️ Without RAG (General Model)")
                st.write(res["norag_answer"])

with tab4:
    st.subheader("📊 Test Set Benchmark & Ablation Study")
    if os.path.exists("EVALUATION_REPORT.md"):
        with open("EVALUATION_REPORT.md", "r", encoding="utf-8") as f:
            report_text = f.read()
        st.markdown(report_text)
    else:
        st.info("Run `python evaluation.py` to generate the evaluation report.")
