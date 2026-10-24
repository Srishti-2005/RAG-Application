# Theme 2B: Open-Source Contributor Guide (Gemma 4 RAG)

A complete Retrieval-Augmented Generation (RAG) system built with **Google Gemma 4** (`gemma-4-26b-a4b-it`) and `sentence-transformers/all-MiniLM-L6-v2`, adapted and extended from the Google AI Studio starter kit for **Hacktoberfest HackDay Theme 2B**.

---

## 🎯 Key Capabilities
1. **Multi-Document & Multi-Format Ingestion**: Ingests `.md`, `.txt`, and `.pdf` files.
2. **Section-Aware Chunking**: Preserves Markdown headers as semantic context anchors.
3. **Cited Answers**: Provides exact citations `(Document, Section)` for every fact.
4. **Transparent Retrieval**: Displays all retrieved passages and similarity scores in an interactive accordion.
5. **Strict Refusal Guardrail**: Automatically declines out-of-domain questions using cosine similarity threshold gating (`SIMILARITY_THRESHOLD = 0.28`) and prompts users toward official community channels (Discussions & Discord).
6. **"Where Do I Start?" 1-Click Onboarding Wizard (2B Bonus)**: Synthesizes a step-by-step setup checklist with verified shell commands directly from docs.
7. **RAG vs No-RAG Experimentation**: Compares grounded responses against general unassisted LLM responses.
8. **18-Question Automated Benchmark Suite**: 100% accuracy on evaluated test set with full ablation study across chunk sizes and top-k values.

---

## 📁 Repository Structure

```
├── 02_rag_starter_kit_extended_2B.ipynb  # Google Colab Ready Notebook
├── rag_engine.py                         # Core modular RAG pipeline & abstractions
├── evaluation.py                         # 18-question benchmark & ablation study
├── EVALUATION_REPORT.md                  # Generated benchmark & ablation report
├── gradio_app.py                         # Full-featured Gradio web interface (Colab/Cloud)
├── streamlit_app.py                      # Streamlit web interface (Local machine)
├── PRESENTATION_GUIDE.md                 # Hackathon showcase presentation guide & script
├── create_notebook.py                    # Generator script for Colab notebook
└── docs/                                 # Official Contributor Documentation Corpus
    ├── README.md                         # Project overview, community links, license
    ├── CONTRIBUTING.md                   # PR workflow, good first issues, translations
    ├── DEVELOPMENT_SETUP.md              # Python versions, venv, pdm, pip, pytest commands
    ├── STYLE_GUIDE.md                    # Ruff formatting, 88 chars, mypy type hints
    └── CODE_OF_CONDUCT.md                # Contributor Covenant v2.0, reporting conduct
```

---

## 🚀 How to Run

### Option 1: Run in Google Colab (Recommended for Hackathon Showcase)
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `02_rag_starter_kit_extended_2B.ipynb`.
3. In Colab Secrets (the key icon in the left sidebar), add `GOOGLE_API_KEY` from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) with Notebook access enabled.
4. Run all cells from top to bottom (Runtime → Run all).
5. Step 7 will launch a public Gradio URL (`https://....gradio.live`) to interact with live!

### Option 2: Run Locally (Local Machine)
1. Ensure dependencies are installed:
   ```bash
   pip install pypdf sentence-transformers gradio google-genai streamlit scikit-learn numpy
   ```
2. (Optional) Set your Google AI Studio API key:
   ```powershell
   $env:GOOGLE_API_KEY="your-api-key-here"
   ```
3. Run the evaluation benchmark:
   ```bash
   python evaluation.py
   ```
4. Launch the local web interface:
   ```bash
   streamlit run streamlit_app.py
   # Or for Gradio:
   python gradio_app.py
   ```

---

## 📊 Benchmark Summary
- **Total Questions**: 18
- **In-Domain Questions**: 14 (All answered with exact document citations)
- **Refusal Test Questions**: 4 (100% correctly refused with community contact info)
- **Overall Accuracy**: **100.0%**
- **Optimal Hyperparameters**: `CHUNK_SIZE = 120 words`, `TOP_K = 3`, `SIMILARITY_THRESHOLD = 0.28`
