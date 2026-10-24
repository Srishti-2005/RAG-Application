"""
gradio_app.py
Full-featured Gradio web application for Theme 2B: Open-Source Contributor Guide.
Designed for Google Colab and local deployment.
"""

import os
import glob
import gradio as gr
from rag_engine import ContributorGuideRAG, REFUSAL_MESSAGE

# Initialize RAG Engine
rag_engine = ContributorGuideRAG(chunk_size=120, chunk_overlap=30, top_k=3, similarity_threshold=0.28)

# Load existing docs on startup if available
default_docs = glob.glob("docs/*.md")
if default_docs:
    try:
        rag_engine.index_files(default_docs)
    except Exception as e:
        print(f"Initial doc indexing warning: {e}")

APP_TITLE = "🛠️ Open-Source Contributor Guide (Gemma 4 RAG)"
APP_SUBTITLE = "Instant answers from official contributor docs with citations, strict refusal guardrails, and newcomer onboarding tools."


def handle_ask(question: str, history: list):
    """Answers contributor questions with source transparency."""
    if not question or not question.strip():
        return "⚠️ Please enter a question in the box above.", "", "⚠️ Missing Question"
        
    answer_text, passages, was_refused = rag_engine.answer(question)
    
    # Format retrieved sources nicely
    if not passages:
        sources_md = "*No passages were retrieved.*"
    else:
        blocks = []
        for p in passages:
            doc_badge = f"📄 `{p['doc']}`"
            sec_badge = f"🏷️ *{p['section']}*"
            score_badge = f"📊 Similarity: **{p['score']:.2f}**"
            blocks.append(
                f"{doc_badge} | {sec_badge} | {score_badge}\n> {p['text']}"
            )
        sources_md = "\n\n---\n\n".join(blocks)
        
    status_label = "⚠️ Refused (Out of Scope)" if was_refused else "✅ Grounded Answer"
    return answer_text, sources_md, status_label


def handle_file_upload(files):
    """Handles multi-file upload (.md, .txt, .pdf)."""
    if not files:
        return "No files selected."
    try:
        file_paths = [f.name if hasattr(f, "name") else str(f) for f in files]
        msg = rag_engine.index_files(file_paths)
        return f"✅ {msg}\n\nIndexed: " + ", ".join([os.path.basename(p) for p in file_paths])
    except Exception as e:
        return f"Error indexing documents: {e}"


def handle_onboarding():
    """Generates the step-by-step onboarding checklist."""
    checklist, passages = rag_engine.generate_onboarding_checklist()
    return checklist


def handle_compare(question: str):
    """RAG vs No-RAG comparison."""
    if not question or not question.strip():
        return "Please enter a question to compare.", "Please enter a question to compare."
    res = rag_engine.compare_rag_vs_norag(question)
    return res["rag_answer"], res["norag_answer"]


def create_ui():
    """Constructs the Gradio user interface."""
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}\n### {APP_SUBTITLE}")
        
        with gr.Tabs():
            # TAB 1: Contributor Q&A
            with gr.TabItem("💬 Ask Contributor Docs"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question_box = gr.Textbox(
                            label="Your Contributor Question",
                            placeholder="e.g., How do I set up my development environment and run tests?",
                            lines=2
                        )
                        with gr.Row():
                            ask_btn = gr.Button("🔍 Ask Assistant", variant="primary")
                            clear_btn = gr.Button("Clear")
                            
                        gr.Examples(
                            examples=[
                                "How do I set up the project on my computer?",
                                "What command should I run to execute the test suite?",
                                "What is the process for submitting a pull request?",
                                "Which coding style rules must I follow for formatting?",
                                "Where can I report violations of the Code of Conduct?",
                                "Can I contribute without writing Python code?",
                                "Who won the 2022 FIFA World Cup in Qatar?",
                                "What is the tuition fee for the semester?"
                            ],
                            inputs=question_box,
                            label="💡 Example Questions (including refusal tests)"
                        )
                        
                        answer_output = gr.Markdown(label="Assistant Answer")
                        status_tag = gr.Markdown()
                        
                        with gr.Accordion("🔎 Retrieved Context Passages (Full Transparency)", open=False):
                            sources_output = gr.Markdown()
                            
                    with gr.Column(scale=1):
                        gr.Markdown("### 📂 Document Knowledge Base")
                        doc_status = gr.Markdown(
                            f"**Currently Loaded Docs**:\n- " + "\n- ".join(rag_engine.knowledge_base["docs"] or ["None loaded"])
                        )
                        upload_box = gr.File(
                            label="Upload Contributor Docs (.md, .txt, .pdf)",
                            file_count="multiple",
                            file_types=[".md", ".txt", ".pdf"]
                        )
                        index_btn = gr.Button("📥 Index Uploaded Files")
                        upload_result = gr.Markdown()
                        
                        load_defaults_btn = gr.Button("🔄 Reload Bundled Docs")

            # TAB 2: Onboarding Wizard (Bonus 2B)
            with gr.TabItem("🚀 Where Do I Start? (Onboarding Wizard)"):
                gr.Markdown(
                    "### 🌟 1-Click Newcomer Onboarding Checklist\n"
                    "First-time contributors often struggle to synthesize setup steps across disparate files. "
                    "Click below to generate a consolidated, verified step-by-step guide directly from the indexed repository docs!"
                )
                generate_checklist_btn = gr.Button("✨ Generate Step-by-Step Onboarding Checklist", variant="primary")
                checklist_output = gr.Markdown()

            # TAB 3: RAG vs No-RAG Comparison
            with gr.TabItem("🔬 RAG vs No-RAG Experiment"):
                gr.Markdown(
                    "### 📊 Hallucination Defense: Comparing Answers With and Without Retrieval\n"
                    "See how standard LLM general knowledge produces generic answers, whereas RAG strictly cites project-specific rules, commands, and linters."
                )
                compare_q = gr.Textbox(
                    label="Test Question for Comparison",
                    value="Which linter and code formatting tool does this project use?"
                )
                compare_btn = gr.Button("⚖️ Compare Side-by-Side", variant="secondary")
                with gr.Row():
                    rag_col = gr.Markdown("#### ✅ With RAG (Grounded in Docs)\n*Click compare to view*")
                    norag_col = gr.Markdown("#### ⚠️ Without RAG (General LLM Memory)\n*Click compare to view*")

            # TAB 4: Architecture & Design Decisions
            with gr.TabItem("ℹ️ Pipeline Architecture & Design"):
                gr.Markdown(
                    """
                    ### 🏗️ RAG Pipeline Architecture
                    
                    ```
                    Multi-Format Docs (.md, .txt, .pdf)
                            │
                            ▼
                    1. Header-Aware Section Parsing (preserves ## titles)
                            │
                            ▼
                    2. Overlapping Chunking (CHUNK_SIZE = 120 words, OVERLAP = 30 words)
                            │
                            ▼
                    3. Embedding Engine (all-MiniLM-L6-v2 / Hybrid Normalization)
                            │
                            ▼
                    Question ──► 4. Cosine Similarity Vector Retrieval (TOP_K = 3)
                            │
                            ▼
                    5. Relevance Threshold Filter (Score < 0.28 ──► Decline & Refuse)
                            │
                            ▼ (Passages >= 0.28)
                    6. Gemma 4 Grounded Generation with Citations & Entity Checking
                    ```
                    
                    #### 🎯 Key Design Decisions:
                    - **Chunk Size = 120 words**: Prevents splitting code blocks while preserving semantic density.
                    - **Top-K = 3**: Optimal coverage without diluting context or increasing Gemma API latency.
                    - **Two-Tier Refusal**: Cosine threshold gating + Gemma system prompt refusal rules.
                    """
                )

        # Wire UI Actions
        ask_btn.click(handle_ask, inputs=[question_box, gr.State([])], outputs=[answer_output, sources_output, status_tag])
        question_box.submit(handle_ask, inputs=[question_box, gr.State([])], outputs=[answer_output, sources_output, status_tag])
        clear_btn.click(lambda: ("", "", "", ""), outputs=[question_box, answer_output, sources_output, status_tag])
        
        index_btn.click(handle_file_upload, inputs=upload_box, outputs=upload_result)
        load_defaults_btn.click(
            lambda: (rag_engine.index_files(glob.glob("docs/*.md")), f"**Currently Loaded Docs**:\n- " + "\n- ".join(rag_engine.knowledge_base["docs"])),
            outputs=[upload_result, doc_status]
        )
        
        generate_checklist_btn.click(handle_onboarding, outputs=checklist_output)
        compare_btn.click(handle_compare, inputs=compare_q, outputs=[rag_col, norag_col])

    return demo


if __name__ == "__main__":
    demo_app = create_ui()
    demo_app.launch(share=False)
