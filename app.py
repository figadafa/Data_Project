import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "Src"))
from Sentimen_Analysis import preprocess_text  # noqa: E402

MODEL_PATH = os.path.join(BASE_DIR, "Output", "best_model.joblib")
TFIDF_PATH = os.path.join(BASE_DIR, "Output", "best_tfidf.joblib")
RESULTS_PATH = os.path.join(BASE_DIR, "Output", "results.csv")
DATA_PATH = os.path.join(BASE_DIR, "Data", "amazon.csv")

# Jarak ke batas keputusan di bawah nilai ini dianggap Netral/Mix (heuristik).
NEUTRAL_MARGIN = 0.3
MAX_INPUT_CHARS = 2000
MAX_BATCH_ROWS = 1000

st.set_page_config(
    page_title="ReviewLens",
    page_icon=os.path.join(BASE_DIR, "asset", "reviewlens-icon.png"),
    layout="centered",
)
st.logo(
    os.path.join(BASE_DIR, "asset", "reviewlens-lockup.png"),
    icon_image=os.path.join(BASE_DIR, "asset", "reviewlens-icon.png"),
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@600;700&display=swap');

    .block-container { max-width: 860px; padding-top: 5rem; padding-bottom: 3rem; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .rl-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
        color: #77736E; text-transform: uppercase;
    }
    .rl-label-accent { color: #F36B35; }

    .rl-topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem; flex-wrap: wrap; gap: 0.5rem; }
    .rl-brand { display: flex; align-items: center; gap: 0.6rem; }
    .rl-brand-square { width: 10px; height: 10px; background: #F36B35; }
    .rl-brand-name { font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: 0.08em; color: #F0EBE2; }

    .rl-notice {
        background: #1c1b1b; padding: 0.6rem 0.9rem; margin-bottom: 1.25rem;
        display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;
    }
    .rl-notice-dot { width: 8px; height: 8px; background: #F36B35; display: inline-block; }

    .rl-card { background: #201f1f; padding: 1rem 1.1rem; margin-top: 0.9rem; }
    .rl-card-inner { background: #0e0e0e; padding: 0.9rem 1rem; }

    .rl-verdict {
        display: inline-block; padding: 0.35rem 0.8rem; font-family: 'JetBrains Mono', monospace;
        font-weight: 700; font-size: 0.9rem; letter-spacing: 0.06em; text-transform: uppercase;
    }
    .rl-pos { background: rgba(78,222,163,0.12); color: #4edea3; }
    .rl-neg { background: rgba(255,180,171,0.12); color: #ffb4ab; }
    .rl-neu { background: rgba(251,191,36,0.12); color: #fbbf24; }

    .rl-metric-num {
        font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 700;
        color: #F0EBE2; letter-spacing: -0.04em; line-height: 1;
    }
    .rl-muted { color: #77736E; font-size: 0.85rem; }

    .rl-word { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; display: flex; justify-content: space-between; padding: 0.2rem 0; color: #F0EBE2; }
    .rl-word-pos { color: #4edea3; }
    .rl-word-neg { color: #ffb4ab; }

    .stButton > button[kind="primary"] {
        background: #F36B35; color: #171717; border: none; border-radius: 0;
        font-family: 'JetBrains Mono', monospace; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    }
    .stButton > button[kind="primary"]:hover { background: #F0EBE2; color: #171717; }
    .stButton > button:not([kind="primary"]) { border-radius: 0; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; text-transform: uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, tfidf


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


def classify(scores: np.ndarray) -> np.ndarray:
    labels = np.where(scores >= 0, "Positif", "Negatif").astype(object)
    labels[np.abs(scores) < NEUTRAL_MARGIN] = "Netral/Mix"
    return labels


def predict_texts(texts, model, tfidf):
    cleaned = [preprocess_text(str(t)) for t in texts]
    X = tfidf.transform(cleaned)
    scores = model.decision_function(X)
    return classify(scores), scores, X


def top_word_contributions(text_vec_row, model, tfidf, top_n=5):
    """Kata dalam ulasan yang paling mendorong prediksi (kontribusi = nilai TF-IDF x koefisien)."""
    feature_names = np.array(tfidf.get_feature_names_out())
    coef = model.coef_[0]
    row = text_vec_row.toarray()[0]
    idx = np.nonzero(row)[0]
    contrib = row[idx] * coef[idx]
    pos_order = np.argsort(-contrib)
    neg_order = np.argsort(contrib)
    pos = [(feature_names[idx[i]], float(contrib[i])) for i in pos_order[:top_n] if contrib[i] > 0]
    neg = [(feature_names[idx[i]], float(contrib[i])) for i in neg_order[:top_n] if contrib[i] < 0]
    return pos, neg


def verdict_html(label: str) -> str:
    css = {"Positif": "rl-pos", "Negatif": "rl-neg"}.get(label, "rl-neu")
    return f'<span class="rl-verdict {css}">{label}</span>'


def word_rows(items, css_class):
    if not items:
        return '<div class="rl-muted">Tidak ada</div>'
    return "".join(
        f'<div class="rl-word"><span>{w}</span><span class="{css_class}">{v:+.3f}</span></div>'
        for w, v in items
    )


model, tfidf = load_artifacts()

# ---------------------------------------------------------------- Header
st.markdown(
    """
    <div class="rl-topbar">
        <div class="rl-brand">
            <span class="rl-brand-square"></span>
            <span class="rl-brand-name">REVIEWLENS</span>
        </div>
        <span class="rl-label">// Sentiment evaluator &middot; TF-IDF + LinearSVC</span>
    </div>
    <div class="rl-notice">
        <span class="rl-notice-dot"></span>
        <span class="rl-label" style="color:#F0EBE2">Catatan: model dilatih hanya dengan ulasan Amazon berbahasa Inggris.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_perf = st.tabs(["01 Satu ulasan", "02 Banyak ulasan", "03 Performa model"])

# ---------------------------------------------------------------- Satu ulasan
with tab_single:
    review = st.text_area(
        "// INPUT_STREAM",
        placeholder="Tulis ulasan dalam bahasa Inggris, misalnya: The battery lasts all day.",
        height=140,
        max_chars=MAX_INPUT_CHARS,
        key="single_text",
    )

    if st.button("Prediksi sentimen", type="primary", key="predict_single"):
        if not review or not review.strip():
            st.warning("Teks ulasan masih kosong. Tulis ulasan terlebih dahulu.")
        else:
            labels, scores, X = predict_texts([review], model, tfidf)
            label, score = labels[0], float(scores[0])
            pos, neg = top_word_contributions(X[0], model, tfidf)

            st.markdown(
                f"""
                <div class="rl-card">
                    <div class="rl-label" style="margin-bottom:0.6rem">// DETECTED_VALENCE</div>
                    {verdict_html(label)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""
                    <div class="rl-card-inner">
                        <div class="rl-label">// DECISION_SCORE</div>
                        <div class="rl-metric-num" style="margin-top:0.4rem">{score:+.3f}</div>
                        <div class="rl-muted" style="margin-top:0.5rem">
                            Jarak ke batas keputusan model, bukan probabilitas.
                            Semakin jauh dari 0, semakin yakin model.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="rl-card-inner">
                        <div class="rl-label">// NEUTRAL_BAND</div>
                        <div class="rl-metric-num" style="margin-top:0.4rem">&plusmn;{NEUTRAL_MARGIN}</div>
                        <div class="rl-muted" style="margin-top:0.5rem">
                            Skor di dalam rentang ini ditandai Netral/Mix. Ini aturan tambahan,
                            karena model asli hanya mengenal positif dan negatif.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            d1, d2 = st.columns(2)
            with d1:
                st.markdown(
                    f"""
                    <div class="rl-card-inner">
                        <div class="rl-label rl-label-accent" style="margin-bottom:0.5rem">Kata pendorong positif</div>
                        {word_rows(pos, "rl-word-pos")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(
                    f"""
                    <div class="rl-card-inner">
                        <div class="rl-label rl-label-accent" style="margin-bottom:0.5rem">Kata pendorong negatif</div>
                        {word_rows(neg, "rl-word-neg")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------- Banyak ulasan
with tab_batch:
    st.markdown(
        '<div class="rl-card"><span class="rl-label rl-label-accent">// BATCH_INGESTION</span>'
        '<div class="rl-muted" style="margin-top:0.4rem">Unggah CSV dengan kolom <code>Text</code>, '
        f"atau pakai 30 ulasan contoh. Maksimal {MAX_BATCH_ROWS} baris.</div></div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Unggah CSV", type=["csv"], key="batch_upload")
    sample_clicked = st.button("Pakai 30 ulasan contoh", key="batch_sample")

    if sample_clicked:
        full = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        st.session_state["batch_source"] = full.sample(n=30, random_state=42).reset_index(drop=True)

    source_df = None
    if uploaded is not None:
        try:
            source_df = pd.read_csv(uploaded, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            st.error(f"File tidak bisa dibaca sebagai CSV: {exc}")
    else:
        source_df = st.session_state.get("batch_source")

    if source_df is not None:
        if "Text" not in source_df.columns:
            st.error(
                "CSV harus memiliki kolom bernama `Text`. Kolom yang ditemukan: "
                + ", ".join(map(str, source_df.columns))
            )
        else:
            df = source_df[["Text"]].copy()
            df["Text"] = df["Text"].fillna("").astype(str)
            df = df[df["Text"].str.strip() != ""].head(MAX_BATCH_ROWS).reset_index(drop=True)

            if df.empty:
                st.warning("Tidak ada ulasan yang bisa dianalisis di file ini.")
            else:
                labels, scores, _ = predict_texts(df["Text"], model, tfidf)
                result = pd.DataFrame(
                    {"Text": df["Text"], "Prediksi": labels, "Skor": np.round(scores, 3)}
                )
                counts = result["Prediksi"].value_counts()
                total = len(result)

                k1, k2, k3 = st.columns(3)
                for col, name in [(k1, "Positif"), (k2, "Negatif"), (k3, "Netral/Mix")]:
                    n = int(counts.get(name, 0))
                    pct = n / total * 100 if total else 0
                    with col:
                        st.markdown(
                            f"""
                            <div class="rl-card-inner">
                                <div class="rl-label">// {name.upper()}</div>
                                <div class="rl-metric-num" style="margin-top:0.4rem">{n:02d}</div>
                                <div class="rl-muted">{pct:.1f}% dari {total} ulasan</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                st.dataframe(result, hide_index=True)
                st.download_button(
                    "Unduh hasil (CSV)",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="reviewlens_hasil.csv",
                    mime="text/csv",
                )

# ---------------------------------------------------------------- Performa model
with tab_perf:
    st.markdown(
        '<div class="rl-card"><span class="rl-label rl-label-accent">// MODEL_EVALUATION</span>'
        '<div class="rl-muted" style="margin-top:0.4rem">Angka di bawah diambil dari '
        "<code>Output/results.csv</code> hasil eksperimen. Dataset tidak seimbang (sekitar 76% positif), "
        "sehingga baseline dummy yang selalu menebak mayoritas sudah mendapat akurasi sekitar 0.76.</div></div>",
        unsafe_allow_html=True,
    )
    st.dataframe(load_results(), hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(os.path.join(BASE_DIR, "Output", "label_distribution.png"), caption="Distribusi label")
    with col_b:
        st.image(os.path.join(BASE_DIR, "Output", "top_words.png"), caption="Kata paling sering muncul")