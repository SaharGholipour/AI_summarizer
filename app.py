"""
AI Summarizer Web App (Streamlit)
------------------------------------
- Search arXiv
- View title/authors/date/PDF link/abstract
- Summarize any paper's abstract (Gemini legacy SDK or local HuggingFace model)

Run:
  pip install streamlit arxiv python-dotenv
  # optional (Gemini legacy SDK):
  pip install google-generativeai
  # optional (local summarizer):
  pip install transformers torch accelerate

  export GEMINI_API_KEY=...
  streamlit run app.py
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
import arxiv
from dotenv import load_dotenv

# =====================
# Environment & storage
# =====================
load_dotenv()
PAPER_DIR = Path("papers")
PAPER_DIR.mkdir(exist_ok=True)

DEFAULT_MAX_RESULTS = 10
DEFAULT_SORT = arxiv.SortCriterion.Relevance


# =====================
# Data model
# =====================
@dataclass
class PaperInfo:
    pid: str
    title: str
    authors: List[str]
    summary: str
    pdf_url: str
    published: str

    def to_row(self) -> dict:
        return {
            "arXiv ID": self.pid,
            "Title": self.title,
            "Authors": ", ".join(self.authors),
            "Published": self.published,
            "PDF": self.pdf_url,
        }


# =====================
# arXiv utilities
# =====================
def _topic_dir(topic: str) -> Path:
    # each topic gets its own subdirectory
    return PAPER_DIR / topic.lower().replace(" ", "_")


def search_papers(topic: str, max_results: int = DEFAULT_MAX_RESULTS,
                  sort_by: arxiv.SortCriterion = DEFAULT_SORT) -> List[PaperInfo]:
    """Search arXiv and cache JSON to papers/<topic>/papers_info.json."""
    client = arxiv.Client()
    results_iter = client.results(
        arxiv.Search(query=topic, max_results=max_results, sort_by=sort_by)
    )

    topic_dir = _topic_dir(topic)
    topic_dir.mkdir(parents=True, exist_ok=True)
    file_path = topic_dir / "papers_info.json"

    try:
        with open(file_path, "r") as f:
            cache = json.load(f)
    except Exception:
        cache = {}

    paper_list: List[PaperInfo] = []
    for paper in results_iter:
        pid = paper.get_short_id()
        info = PaperInfo(
            pid=pid,
            title=paper.title,
            authors=[a.name for a in paper.authors],
            summary=paper.summary,
            pdf_url=paper.pdf_url,
            published=str(paper.published.date()),
        )
        cache[pid] = asdict(info)
        paper_list.append(info)

    with open(file_path, "w") as f:
        json.dump(cache, f, indent=2)

    return paper_list


def extract_info_by_id(paper_id: str) -> Optional[PaperInfo]:
    """Load a single paper from any topic cache by its arXiv ID."""
    for subdir in PAPER_DIR.iterdir():
        if subdir.is_dir():
            file_path = subdir / "papers_info.json"
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    if paper_id in data:
                        d = data[paper_id]
                        return PaperInfo(
                            pid=paper_id,
                            title=d["title"],
                            authors=d["authors"],
                            summary=d["summary"],
                            pdf_url=d["pdf_url"],
                            published=d["published"],
                        )
                except Exception:
                    continue
    return None


# “Quick view” can fetch a paper by ID even if we never searched that topic, then it saves it for future runs.
def fetch_and_cache_by_id(paper_id: str) -> Optional[PaperInfo]:
    """Query arXiv by ID and cache it so extract_info_by_id can find it later."""
    base = paper_id.replace("arXiv:", "").split("v")[0]
    client = arxiv.Client()
    results = list(client.results(arxiv.Search(id_list=[base])))
    if not results:
        return None

    p = results[0]
    info = PaperInfo(
        pid=p.get_short_id(),
        title=p.title,
        authors=[a.name for a in p.authors],
        summary=p.summary,
        pdf_url=p.pdf_url,
        published=str(p.published.date()),
    )

    by_id_dir = PAPER_DIR / "_by_id"
    by_id_dir.mkdir(parents=True, exist_ok=True)
    fp = by_id_dir / "papers_info.json"

    try:
        with open(fp, "r") as f:
            cache = json.load(f)
    except Exception:
        cache = {}

    cache[info.pid] = asdict(info)
    with open(fp, "w") as f:
        json.dump(cache, f, indent=2)

    return info


# =====================
# Summarizers
# =====================
class Summarizer:
    def summarize(self, text: str, hint: str = "academic paper") -> str:
        raise NotImplementedError


class GeminiLegacySummarizer(Summarizer):
    """google-generativeai (legacy)"""
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "google-generativeai is not installed.\n"
                "Install with: pip install -U google-generativeai"
            ) from e
        self._genai = genai
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY for Gemini summarizer.")
        self._genai.configure(api_key=self.api_key)
        self.model = self._genai.GenerativeModel(
            model,
            system_instruction="You are a helpful research assistant.",
        )

    def summarize(self, text: str, hint: str = "academic paper") -> str:
        prompt = (
            f"Summarize the following {hint} for a technical audience in ≤200 words.\n"
            "Highlight: problem, method, key results, limitations.\n\n"
            f"TEXT:\n{text}"
        )
        resp = self.model.generate_content(prompt, generation_config={"max_output_tokens": 500})
        out = getattr(resp, "text", None)
        if out:
            return out.strip()
        # fallback assembly from candidates
        parts: List[str] = []
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            if content:
                for p in getattr(content, "parts", []) or []:
                    t = getattr(p, "text", None)
                    if t:
                        parts.append(t)
        return ("\n".join(parts)).strip()


class HFLocalSummarizer(Summarizer):
    """Local HuggingFace summarizer with chunked map-reduce."""
    def __init__(self, model_id: str = "sshleifer/distilbart-cnn-12-6"):
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "transformers not installed.\n"
                "Install with: pip install transformers torch accelerate"
            ) from e
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
        self.max_chars = 3000

    def _summ_once(self, txt: str) -> str:
        out = self.pipe(txt, max_length=220, min_length=80, do_sample=False)
        return out[0]["summary_text"].strip()

    def summarize(self, text: str, hint: str = "academic paper") -> str:
        t = text.strip()
        if len(t) <= self.max_chars:
            return self._summ_once(t)
        chunks = _split_text(t, chunk_size=self.max_chars, overlap=400)
        partials = [self._summ_once(c) for c in chunks]
        merged = "\n".join(partials)
        return self._summ_once("Summaries of sections:\n" + merged + "\n\nCreate a concise, unified abstract.")


def _split_text(s: str, chunk_size: int = 3000, overlap: int = 400) -> List[str]:
    s = " ".join(s.split())  # normalize whitespace
    parts: List[str] = []
    i = 0
    while i < len(s):
        parts.append(s[i : i + chunk_size])
        i += chunk_size - overlap
        if i < 0:  # safety
            break
    return parts


# =====================
# Streamlit UI helpers
# =====================
@st.cache_data(show_spinner=False)
def cached_search(topic: str, max_results: int) -> List[PaperInfo]:
    return search_papers(topic, max_results=max_results)


def build_summarizer(choice: str, model_name: str) -> Summarizer:
    if choice == "Gemini (legacy SDK)":
        return GeminiLegacySummarizer(model=model_name)
    else:
        return HFLocalSummarizer(model_id=model_name)


# =====================
# App
# =====================
st.set_page_config(page_title="AI Summarizer", layout="wide")

# Persistent UI state (keeps results visible across reruns)
if "papers" not in st.session_state:
    st.session_state["papers"] = []
    st.session_state["topic"] = ""
    st.session_state["id_options"] = []
    st.session_state["selected_id"] = None


st.title("📄 AI Summarizer")
st.caption("Search papers, inspect abstracts, and generate short technical summaries.")

with st.sidebar:
    st.header("Settings")
    provider = st.radio(
        "Summarizer backend",
        ["Gemini (legacy SDK)", "HF (HuggingFace)"],
        help="Gemini uses google-generativeai with GEMINI_API_KEY. HF uses a Transformers model on your machine.",
    )
    if provider == "Gemini (legacy SDK)":
        model_name = st.text_input("Gemini model", value="gemini-1.5-flash")
        st.info("Install: pip install google-generativeai · Set GEMINI_API_KEY in .env .")
    else:
        model_name = st.text_input("HF model", value="sshleifer/distilbart-cnn-12-6")
        st.info("Install: pip install transformers torch accelerate (CPU works; GPU faster)")

    st.divider()
    with st.expander("Tips"):
        st.markdown(
            "- Use Enter or click Search arXiv to run.\n"
            "- Click a row to copy the arXiv ID; paste it below to summarize.\n"
            "- HF models are fully free; Gemini free tier has rate limits."
        )

# Search form
with st.form("search_form"):
    col_left, col_right = st.columns([1.3, 1])
    with col_left:
        topic = st.text_input("Search topic", placeholder="e.g., B meson decays", key="topic_input")
    with col_right:
        max_results = st.slider("Number of papers", 1, 10, 3, key="number_of_papers")
    submitted = st.form_submit_button("🔎 Search arXiv", use_container_width=True)

if submitted:
    if topic and topic.strip():
        with st.spinner("Searching arXiv..."):
            papers = cached_search(topic.strip(), max_results=max_results)
        st.session_state["papers"] = papers
        st.session_state["topic"] = topic.strip()
        st.session_state["id_options"] = [p.pid for p in papers]
        st.session_state["selected_id"] = None
        st.success(f"Found {len(papers)} results for '{topic}'.")
    else:
        st.warning("Please enter a topic.")

# Render results from session state so they persist on reruns
papers = st.session_state.get("papers", [])
if papers:
    rows = [p.to_row() for p in papers]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Quick view")

    # Option A: select from current results
    def _on_pick_change():
        st.session_state["selected_id"] = st.session_state.get("pick_from_results")

    st.selectbox(
        "Click an ID to preview/summarize",
        options=st.session_state.get("id_options", []),
        index=None,
        placeholder="Select a result…",
        key="pick_from_results",
        on_change=_on_pick_change,
    )

    # Option B: or paste an ID (on_change stores into session)
    def _set_selected_from_input():
        val = (st.session_state.get("id_input") or "").strip()
        if val:
            st.session_state["selected_id"] = val

    st.text_input(
        "Or paste an arXiv ID",
        placeholder="e.g., 1904.12858v4",
        key="id_input",
        on_change=_set_selected_from_input,
    )

    # Determine the current selection from session state
    selected_id = st.session_state.get("selected_id")
    if selected_id:
        st.session_state["selected_id"] = selected_id
        info = extract_info_by_id(selected_id)
        if not info:
            with st.spinner("Fetching paper by arXiv ID…"):
                info = fetch_and_cache_by_id(selected_id)
        if info:
            st.markdown(f"### {info.title}")
            st.write(f"**Authors:** {', '.join(info.authors)}")
            st.write(f"**Published:** {info.published}")
            st.write(f"**PDF:** {info.pdf_url}")
            with st.expander("Abstract"):
                st.write(info.summary)
            # Sync into summarization input
            st.session_state["sum_id"] = selected_id
        else:
            st.error("Could not find that arXiv ID.")

# Summarization section
st.divider()
st.subheader("Summarize a paper or any text")
input_mode = st.radio("Input", ["Use arXiv ID (from above)", "Paste custom text"], horizontal=True)

summ_text: Optional[str] = None
if input_mode == "Use arXiv ID (from above)":
    pid_for_sum = st.text_input("arXiv ID", key="sum_id")
    if pid_for_sum:
        info = extract_info_by_id(pid_for_sum.strip())
        if info:
            summ_text = info.summary
            st.caption("Loaded abstract from cache. Click Summarize.")
        else:
            st.warning("ID not found in cache. Search the topic first, or paste text instead.")
else:
    summ_text = st.text_area("Text to summarize", height=200, placeholder="Paste abstract or long paragraph...")

col_a, col_b = st.columns([1, 1])
with col_a:
    do_sum = st.button("🧠 Summarize", type="primary")
with col_b:
    hint = st.text_input("Domain hint", value="academic paper", help="Used to guide the summary style.")

if do_sum:
    if not summ_text or not summ_text.strip():
        st.error("Please provide arXiv ID or paste text.")
    else:
        try:
            summarizer = build_summarizer(provider, model_name)
            with st.spinner(f"Summarizing with {provider}..."):
                out = summarizer.summarize(summ_text, hint=hint)
            st.success("Done.")
            st.markdown("### Summary")
            st.write(out)
        except Exception as e:
            st.exception(e)

# Footer
st.divider()
st.caption("AI Summarizer • Built with Streamlit · arxiv · google-generativeai/Transformers")
