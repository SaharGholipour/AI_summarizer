# Summarizer Web App (Streamlit)
Search arXiv, preview papers, and generate short technical summaries. Choose between a free **local HuggingFace** summarizer or **Google Gemini (legacy SDK)** if you have an API key.

## Features
- 🔎 **Search arXiv** and cache results locally
- 📑 **Quick view**: title · authors · date · direct **PDF** link · abstract
- 🧠 **Summarize** abstracts using:
  - **Gemini (legacy)** via `google-generativeai`, or
  - **Local** models via 🤗 Transformers (free, runs on CPU/GPU)

---

## Run
- **streamlit run app.py**
