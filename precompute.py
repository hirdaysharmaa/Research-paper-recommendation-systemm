#!/usr/bin/env python3
"""
Pre-builds TF-IDF and SBERT+FAISS caches so the Streamlit app
starts instantly on first use.
"""

import os, pickle, time, pandas as pd, numpy as np, ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss

DATA_PATH   = "arxiv_data.csv"
CACHE_DIR   = "cache"
TFIDF_CACHE = os.path.join(CACHE_DIR, "tfidf_model.pkl")
SBERT_CACHE = os.path.join(CACHE_DIR, "sbert_index.pkl")
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 60)
print("  ArXiv Paper Recommender — Pre-build Caches")
print("=" * 60)

# ── Load Data ────────────────────────────────────────────────
print("\n[1/4] Loading dataset...")
df = pd.read_csv(DATA_PATH)
df.dropna(subset=["titles", "summaries", "terms"], inplace=True)
df.reset_index(drop=True, inplace=True)

def parse_terms(t):
    try:
        p = ast.literal_eval(t)
        return p if isinstance(p, list) else [t]
    except Exception:
        return [str(t)]

df["terms_list"] = df["terms"].apply(parse_terms)
df["combined"]   = df["titles"] + " " + df["summaries"]
print(f"  Loaded {len(df):,} papers.")

# ── TF-IDF ───────────────────────────────────────────────────
if os.path.exists(TFIDF_CACHE):
    print("\n[2/4] TF-IDF cache already exists — skipping.")
else:
    print("\n[2/4] Building TF-IDF index...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=25000, stop_words="english",
        ngram_range=(1, 2), sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(df["combined"].tolist())
    with open(TFIDF_CACHE, "wb") as f:
        pickle.dump((vectorizer, matrix), f)
    print(f"  Done in {time.time()-t0:.1f}s — saved to {TFIDF_CACHE}")

# ── SBERT + FAISS ────────────────────────────────────────────
if os.path.exists(SBERT_CACHE):
    print("\n[3/4] SBERT+FAISS cache already exists — skipping.")
else:
    print("\n[3/4] Encoding papers with SBERT (this takes ~5-10 min for 51K papers)...")
    t0 = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["combined"].tolist(),
        batch_size=256,
        show_progress_bar=True,
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    with open(SBERT_CACHE, "wb") as f:
        pickle.dump((embeddings, index), f)
    print(f"  Done in {time.time()-t0:.1f}s — saved to {SBERT_CACHE}")

# ── Done ─────────────────────────────────────────────────────
print("\n[4/4] All caches ready!")
print("=" * 60)
print("  Run the app with:  streamlit run Updatted app.py")
print("=" * 60)
