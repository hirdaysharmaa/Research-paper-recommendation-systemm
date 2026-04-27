import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import ast
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=" ResearchPaper Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #7c3aed 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(37,99,235,0.3);
    }
    .main-header h1 { font-size: 2.6rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { font-size: 1.1rem; opacity: 0.85; margin-top: 0.5rem; }

    .paper-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #2563eb;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s;
    }
    .paper-card:hover { box-shadow: 0 6px 20px rgba(37,99,235,0.15); }

    .paper-rank {
        display: inline-block;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        font-weight: 700;
        border-radius: 50%;
        width: 32px; height: 32px;
        line-height: 32px;
        text-align: center;
        font-size: 0.9rem;
        margin-right: 0.7rem;
        flex-shrink: 0;
    }
    .paper-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .paper-summary {
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 0.7rem;
    }
    .tag {
        display: inline-block;
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 2px 3px 2px 0;
    }
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        float: right;
    }

    .metric-box {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-box .val { font-size: 1.8rem; font-weight: 700; color: #0369a1; }
    .metric-box .lbl { font-size: 0.8rem; color: #64748b; margin-top: 2px; }

    .method-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px;
        cursor: pointer;
    }
    .active-pill   { background:#2563eb; color:white; }
    .inactive-pill { background:#f1f5f9; color:#64748b; border:1px solid #e2e8f0; }

    .stTextArea textarea {
        border-radius: 10px !important;
        border: 2px solid #e2e8f0 !important;
        font-size: 0.9rem !important;
    }
    .stTextArea textarea:focus { border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important; }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

    .sidebar-section {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }

    div[data-testid="stExpander"] { border-radius: 10px !important; border: 1px solid #e2e8f0 !important; }
    .stSpinner > div { border-top-color: #2563eb !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
DATA_PATH   = "arxiv_data.csv"
CACHE_DIR   = "cache"
TFIDF_CACHE = os.path.join(CACHE_DIR, "tfidf_model.pkl")
SBERT_CACHE = os.path.join(CACHE_DIR, "sbert_index.pkl")
os.makedirs(CACHE_DIR, exist_ok=True)

ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision",            "cs.CL": "Computation & Language",
    "cs.NE": "Neural & Evolutionary",      "cs.RO": "Robotics",
    "cs.DS": "Data Structures",            "cs.SE": "Software Engineering",
    "stat.ML": "Statistics - ML",          "math.OC": "Optimization & Control",
    "cs.IR": "Information Retrieval",      "cs.HC": "Human-Computer Interaction",
}

# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["titles", "summaries", "terms"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    def parse_terms(t):
        try:
            parsed = ast.literal_eval(t)
            return parsed if isinstance(parsed, list) else [t]
        except Exception:
            return [str(t)]

    df["terms_list"] = df["terms"].apply(parse_terms)
    df["combined"]   = df["titles"] + " " + df["summaries"]
    return df

# ─── TF-IDF Model ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tfidf(df):
    if os.path.exists(TFIDF_CACHE):
        with open(TFIDF_CACHE, "rb") as f:
            vectorizer, matrix = pickle.load(f)
    else:
        vectorizer = TfidfVectorizer(
            max_features=25000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(df["combined"].tolist())
        with open(TFIDF_CACHE, "wb") as f:
            pickle.dump((vectorizer, matrix), f)
    return vectorizer, matrix

# ─── SBERT + FAISS Model ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sbert(df):
    if os.path.exists(SBERT_CACHE):
        with open(SBERT_CACHE, "rb") as f:
            embeddings, index = pickle.load(f)
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = df["combined"].tolist()
        embeddings = model.encode(texts, batch_size=256, show_progress_bar=False)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        with open(SBERT_CACHE, "wb") as f:
            pickle.dump((embeddings, index), f)
    return embeddings, index

# ─── Recommendation Engines ──────────────────────────────────────────────────
def recommend_tfidf(query, df, vectorizer, matrix, top_k=10, category_filter=None):
    q_vec  = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    if category_filter:
        for i, row in df.iterrows():
            if not any(c in row["terms_list"] for c in category_filter):
                scores[i] = -1
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if scores[idx] > 0:
            results.append({
                "rank":    len(results) + 1,
                "title":   df.iloc[idx]["titles"],
                "summary": df.iloc[idx]["summaries"],
                "terms":   df.iloc[idx]["terms_list"],
                "score":   round(float(scores[idx]) * 100, 2),
            })
    return results

def recommend_sbert(query, df, embeddings, faiss_index, top_k=10, category_filter=None):
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb  = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    k_fetch = top_k * 5 if category_filter else top_k
    scores, indices = faiss_index.search(q_emb, k_fetch)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if category_filter:
            row = df.iloc[idx]
            if not any(c in row["terms_list"] for c in category_filter):
                continue
        results.append({
            "rank":    len(results) + 1,
            "title":   df.iloc[idx]["titles"],
            "summary": df.iloc[idx]["summaries"],
            "terms":   df.iloc[idx]["terms_list"],
            "score":   round(float(score) * 100, 2),
        })
        if len(results) >= top_k:
            break
    return results

def recommend_hybrid(query, df, vectorizer, tfidf_matrix, embeddings, faiss_index,
                     top_k=10, alpha=0.5, category_filter=None):
    # TF-IDF scores (normalised)
    q_vec   = vectorizer.transform([query])
    tfidf_s = cosine_similarity(q_vec, tfidf_matrix).flatten()
    tfidf_s = (tfidf_s - tfidf_s.min()) / (tfidf_s.max() - tfidf_s.min() + 1e-9)

    # SBERT scores
    model   = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb   = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    sbert_s = np.zeros(len(df))
    raw_scores, indices = faiss_index.search(q_emb, len(df))
    for score, idx in zip(raw_scores[0], indices[0]):
        sbert_s[idx] = score
    sbert_s = (sbert_s - sbert_s.min()) / (sbert_s.max() - sbert_s.min() + 1e-9)

    combined = alpha * sbert_s + (1 - alpha) * tfidf_s

    if category_filter:
        for i, row in df.iterrows():
            if not any(c in row["terms_list"] for c in category_filter):
                combined[i] = -1

    top_idx = np.argsort(combined)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if combined[idx] > 0:
            results.append({
                "rank":    len(results) + 1,
                "title":   df.iloc[idx]["titles"],
                "summary": df.iloc[idx]["summaries"],
                "terms":   df.iloc[idx]["terms_list"],
                "score":   round(float(combined[idx]) * 100, 2),
            })
    return results

# ─── Paper Card HTML ─────────────────────────────────────────────────────────
def render_card(paper):
    tags_html  = "".join(f'<span class="tag">{t}</span>' for t in paper["terms"])
    summary    = paper["summary"][:350] + "..." if len(paper["summary"]) > 350 else paper["summary"]
    return f"""
    <div class="paper-card">
        <div style="display:flex; align-items:flex-start; gap:0.5rem;">
            <div>
                <span class="score-badge">⚡ {paper['score']}%</span>
                <span class="paper-rank">{paper['rank']}</span>
                <span class="paper-title">{paper['title']}</span>
                <div class="paper-summary">{summary}</div>
                <div>{tags_html}</div>
            </div>
        </div>
    </div>"""

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    method = st.selectbox(
        "🔍 Recommendation Method",
        ["🧠 Semantic (SBERT + FAISS)", "📊 TF-IDF (Keyword)", "🔀 Hybrid (Best of Both)"],
        help="Choose the algorithm powering recommendations.",
    )

    top_k = st.slider("📋 Number of Results", 3, 20, 8)

    if "Hybrid" in method:
        alpha = st.slider("⚖️ Semantic Weight (α)", 0.0, 1.0, 0.6, 0.05,
                          help="Higher = more semantic; lower = more keyword-based.")
    else:
        alpha = 0.5

    st.markdown("---")
    st.markdown("### 🏷️ Filter by Category")
    all_cats = list(ARXIV_CATEGORIES.keys())
    selected_cats = st.multiselect(
        "arXiv Categories",
        options=all_cats,
        format_func=lambda x: f"{x} — {ARXIV_CATEGORIES.get(x, x)}",
        help="Leave empty to search across all categories.",
    )

    st.markdown("---")
    st.markdown("### 📌 Quick Queries")
    quick_queries = [
        "transformer attention mechanism NLP",
        "graph neural networks drug discovery",
        "reinforcement learning robotics",
        "diffusion models image generation",
        "federated learning privacy",
    ]
    for q in quick_queries:
        if st.button(f"🔎 {q[:40]}", key=q):
            st.session_state["query"] = q

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **ArXiv Paper Recommender**  
    Final Year Project  
    
    - 📚 51,774 arXiv papers  
    - 🤖 3 AI-powered methods  
    - ⚡ FAISS for fast search  
    - 🧠 SBERT semantic embeddings  
    """)

# ─── Main Layout ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📚 ArXiv Research Paper Recommender</h1>
    <p>Discover relevant research papers using Semantic AI, TF-IDF, and Hybrid approaches</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("📂 Loading dataset..."):
    df = load_data()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-box"><div class="val">{len(df):,}</div><div class="lbl">Total Papers</div></div>', unsafe_allow_html=True)
with col2:
    unique_cats = set(c for tl in df["terms_list"] for c in tl)
    st.markdown(f'<div class="metric-box"><div class="val">{len(unique_cats)}</div><div class="lbl">Categories</div></div>', unsafe_allow_html=True)
with col3:
    method_label = "SBERT" if "Semantic" in method else ("TF-IDF" if "TF-IDF" in method else "Hybrid")
    st.markdown(f'<div class="metric-box"><div class="val">{method_label}</div><div class="lbl">Active Method</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-box"><div class="val">{top_k}</div><div class="lbl">Results per Query</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Model Loading ────────────────────────────────────────────────────────────
tabs = st.tabs(["🔍 Search & Recommend"])

with tabs[0]:
    st.markdown("#### 💬 Enter your research query")
    query = st.text_area(
        "Describe the topic you're looking for:",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="e.g. attention mechanism for long document summarisation using transformers...",
        key="main_query",
    )

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        search_btn = st.button("🚀 Find Papers", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    with col_btn3:
        export_btn = st.button("📥 Export CSV", use_container_width=True)

    if clear_btn:
        st.session_state["query"] = ""
        st.rerun()

    if search_btn and query.strip():
        start_time = time.time()

        if "Semantic" in method or "Hybrid" in method:
            with st.spinner("🧠 Loading Semantic Model & FAISS Index (first run may take a minute)..."):
                embeddings, faiss_index = load_sbert(df)

        if "TF-IDF" in method or "Hybrid" in method:
            with st.spinner("📊 Building TF-IDF Index..."):
                vectorizer, tfidf_matrix = load_tfidf(df)

        with st.spinner(f"🔍 Running {method_label} recommendations..."):
            if "Semantic" in method:
                results = recommend_sbert(query, df, embeddings, faiss_index,
                                          top_k=top_k, category_filter=selected_cats or None)
            elif "TF-IDF" in method:
                results = recommend_tfidf(query, df, vectorizer, tfidf_matrix,
                                          top_k=top_k, category_filter=selected_cats or None)
            else:
                results = recommend_hybrid(query, df, vectorizer, tfidf_matrix,
                                           embeddings, faiss_index, top_k=top_k,
                                           alpha=alpha, category_filter=selected_cats or None)

        elapsed = round(time.time() - start_time, 2)
        st.session_state["last_results"] = results

        if results:
            st.success(f"✅ Found **{len(results)} papers** in **{elapsed}s** using *{method_label}*")
            for paper in results:
                st.markdown(render_card(paper), unsafe_allow_html=True)
        else:
            st.warning("⚠️ No results found. Try a different query or remove category filters.")

    elif search_btn:
        st.warning("⚠️ Please enter a search query!")

    # Export
    if export_btn:
        res = st.session_state.get("last_results", [])
        if res:
            exp_df = pd.DataFrame(res)
            st.download_button(
                "⬇️ Download Results",
                exp_df.to_csv(index=False).encode("utf-8"),
                "recommendations.csv",
                "text/csv",
            )
        else:
            st.info("Run a search first to export results.")
