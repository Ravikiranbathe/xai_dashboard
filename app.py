"""
Explainable AI Dashboard — app.py
Main Streamlit entry point.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="XAI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root & body ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
  background: radial-gradient(ellipse at top left, #1a0533 0%, #0d0d1a 40%, #001a2e 100%);
  min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d0d1a 0%, #12122a 100%);
  border-right: 1px solid rgba(126,255,245,0.12);
}
section[data-testid="stSidebar"] .stRadio label {
  color: #c0c0d8 !important;
  font-size: 0.92rem;
  padding: 0.35rem 0;
  transition: color 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover { color: #7efff5 !important; }

/* ── Logo area ── */
.sidebar-logo {
  text-align: center;
  padding: 1.2rem 0 0.5rem;
}
.sidebar-logo .logo-icon {
  font-size: 2.8rem;
  display: block;
  filter: drop-shadow(0 0 12px #7efff5aa);
}
.sidebar-logo .logo-title {
  font-size: 1.1rem;
  font-weight: 700;
  background: linear-gradient(90deg, #7efff5, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 0.04em;
}
.sidebar-logo .logo-sub {
  font-size: 0.72rem;
  color: #5a5a7a;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ── Pipeline status pills ── */
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.78rem;
  padding: 0.25rem 0.7rem;
  border-radius: 20px;
  margin: 0.18rem 0;
  width: 100%;
}
.status-pill.done {
  background: rgba(74,222,128,0.12);
  border: 1px solid rgba(74,222,128,0.3);
  color: #86efac;
}
.status-pill.pending {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  color: #4a4a6a;
}

/* ── Page header ── */
.page-header {
  padding: 0.5rem 0 1.2rem;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  margin-bottom: 1.5rem;
}
.page-header .page-title {
  font-size: 2rem;
  font-weight: 700;
  background: linear-gradient(90deg, #7efff5 0%, #a78bfa 60%, #f472b6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.2;
  margin: 0;
}
.page-header .page-sub {
  color: #6b6b8a;
  font-size: 0.88rem;
  margin-top: 0.3rem;
}

/* ── Hero (Home) ── */
.hero {
  background: linear-gradient(135deg, rgba(126,255,245,0.06) 0%, rgba(167,139,250,0.06) 100%);
  border: 1px solid rgba(126,255,245,0.1);
  border-radius: 20px;
  padding: 2.5rem 2rem;
  text-align: center;
  margin-bottom: 2rem;
}
.hero h1 {
  font-size: 2.6rem;
  font-weight: 800;
  background: linear-gradient(90deg, #7efff5, #a78bfa, #f472b6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0 0.6rem;
}
.hero p { color: #8888aa; font-size: 1rem; max-width: 560px; margin: 0 auto; }

/* ── Stat cards ── */
.stat-grid { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.stat-card {
  flex: 1;
  min-width: 120px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 1.2rem 1rem;
  text-align: center;
  transition: border-color 0.25s, transform 0.2s;
}
.stat-card:hover {
  border-color: rgba(126,255,245,0.35);
  transform: translateY(-2px);
}
.stat-card .stat-val {
  font-size: 2rem;
  font-weight: 700;
  background: linear-gradient(135deg, #7efff5, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: block;
}
.stat-card .stat-label { color: #5a5a7a; font-size: 0.78rem; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Step cards (Home how-to) ── */
.steps { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; }
.step-card {
  flex: 1;
  min-width: 160px;
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 14px;
  padding: 1.2rem 1rem;
  position: relative;
  transition: border-color 0.25s;
}
.step-card:hover { border-color: rgba(167,139,250,0.4); }
.step-card .step-num {
  font-size: 0.7rem;
  font-weight: 700;
  color: #a78bfa;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
.step-card .step-icon { font-size: 1.6rem; margin: 0.4rem 0; display: block; }
.step-card .step-title { font-size: 0.92rem; font-weight: 600; color: #e0e0f0; }
.step-card .step-desc { font-size: 0.78rem; color: #5a5a7a; margin-top: 0.3rem; }

/* ── Glass panel ── */
.glass-panel {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 1.4rem 1.6rem;
  margin: 1rem 0;
}

/* ── Info / alert boxes ── */
.info-box {
  background: rgba(126,255,245,0.06);
  border-left: 3px solid #7efff5;
  border-radius: 0 8px 8px 0;
  padding: 0.65rem 1rem;
  color: #a8f5f0;
  font-size: 0.88rem;
  margin: 0.6rem 0;
}
.success-box {
  background: rgba(74,222,128,0.07);
  border-left: 3px solid #4ade80;
  border-radius: 0 8px 8px 0;
  padding: 0.65rem 1rem;
  color: #86efac;
  font-size: 0.88rem;
  margin: 0.6rem 0;
}
.warn-box {
  background: rgba(251,146,60,0.07);
  border-left: 3px solid #fb923c;
  border-radius: 0 8px 8px 0;
  padding: 0.65rem 1rem;
  color: #fdba74;
  font-size: 0.88rem;
  margin: 0.6rem 0;
}
.error-box {
  background: rgba(248,113,113,0.07);
  border-left: 3px solid #f87171;
  border-radius: 0 8px 8px 0;
  padding: 0.65rem 1rem;
  color: #fca5a5;
  font-size: 0.88rem;
  margin: 0.6rem 0;
}

/* ── Big result card (Predictions) ── */
.result-card {
  background: linear-gradient(135deg, rgba(126,255,245,0.08), rgba(167,139,250,0.08));
  border: 1px solid rgba(126,255,245,0.2);
  border-radius: 18px;
  padding: 1.8rem;
  text-align: center;
}
.result-card .result-label { font-size: 0.75rem; color: #6b6b8a; text-transform: uppercase; letter-spacing: 0.1em; }
.result-card .result-value {
  font-size: 2.4rem;
  font-weight: 800;
  background: linear-gradient(90deg, #7efff5, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0.2rem 0;
}
.result-card .result-sub { font-size: 0.82rem; color: #5a5a7a; }

/* ── Metric badge ── */
.metric-badge {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 12px;
  padding: 1rem;
  text-align: center;
}
.metric-badge .mb-val { font-size: 1.5rem; font-weight: 700; color: #7efff5; }
.metric-badge .mb-label { font-size: 0.75rem; color: #5a5a7a; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Section divider ── */
.section-divider {
  border: none;
  border-top: 1px solid rgba(255,255,255,0.06);
  margin: 1.5rem 0;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.02) !important;
  border: 2px dashed rgba(126,255,245,0.2) !important;
  border-radius: 14px !important;
  transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(126,255,245,0.5) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(90deg, #7efff5, #a78bfa) !important;
  color: #0d0d1a !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.55rem 1.8rem !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.02em;
  transition: opacity 0.2s, transform 0.15s !important;
  box-shadow: 0 4px 20px rgba(126,255,245,0.2) !important;
}
.stButton > button:hover {
  opacity: 0.88 !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs / selects ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stSlider { color: #e0e0f0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.03);
  border-radius: 10px;
  padding: 0.2rem;
  gap: 0.2rem;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  color: #6b6b8a !important;
  font-size: 0.88rem;
  padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
  background: rgba(126,255,245,0.12) !important;
  color: #7efff5 !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
iframe[title="st_aggrid"] { border-radius: 10px; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: rgba(255,255,255,0.03) !important;
  border-radius: 8px !important;
  color: #a0a0c0 !important;
}

/* ── Metrics (native st.metric) ── */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px;
  padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] { color: #7efff5 !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #5a5a7a !important; font-size: 0.78rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(126,255,245,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(126,255,245,0.4); }
</style>
""", unsafe_allow_html=True)

# ── Imports from utils ─────────────────────────────────────────────────────
from utils.data_processing import (
    handle_missing_values, encode_features,
    scale_features, get_feature_summary,
)
from utils.train_model import (
    split_data, train_all_models,
    plot_confusion_matrix, plot_roc_curve, plot_model_comparison,
)
from utils.explain_model import (
    compute_shap_values, plot_shap_summary,
    plot_shap_waterfall, plot_feature_importance,
    generate_plain_language_explanation,
)

# ── Session state ──────────────────────────────────────────────────────────
_STATE_KEYS = [
    "df", "df_processed", "X", "y", "encoders", "scaler",
    "feature_names", "target_col", "cat_cols", "num_cols",
    "models", "results", "shap_values", "X_test_arr",
    "X_train_arr", "y_test", "encoding_strategy",
]
for _k in _STATE_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None

# ── Helpers ────────────────────────────────────────────────────────────────
def _status(done: bool, label: str) -> str:
    cls = "done" if done else "pending"
    icon = "✦" if done else "○"
    return (f'<div class="status-pill {cls}">'
            f'<span>{icon}</span><span>{label}</span></div>')

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <span class="logo-icon">🧠</span>
      <div class="logo-title">XAI Dashboard</div>
      <div class="logo-sub">Explainable AI Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:0.8rem 0'>",
                unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠  Home", "📂  Upload Data", "🏋️  Train Model",
         "🔮  Predictions", "🔍  Explain Model"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:0.8rem 0'>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#3a3a5a;text-transform:uppercase;"
                "letter-spacing:0.1em;margin-bottom:0.5rem'>Pipeline</div>",
                unsafe_allow_html=True)

    st.markdown(
        _status(st.session_state.df is not None,       "Data loaded") +
        _status(st.session_state.X is not None,        "Data processed") +
        _status(st.session_state.models is not None,   "Models trained") +
        _status(st.session_state.shap_values is not None, "SHAP computed"),
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:0.8rem 0'>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#2a2a4a;text-align:center'>"
                "Built with Streamlit · SHAP · scikit-learn</div>",
                unsafe_allow_html=True)

# ── Page header helper ─────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-title">{title}</div>'
        f'<div class="page-sub">{subtitle}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div class="hero">
      <h1>Explainable AI Dashboard</h1>
      <p>Upload any tabular dataset, train four ML models in one click, compare
         performance metrics, and understand every prediction through SHAP.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
      <div class="stat-card"><span class="stat-val">4</span><div class="stat-label">ML Models</div></div>
      <div class="stat-card"><span class="stat-val">SHAP</span><div class="stat-label">Explainability</div></div>
      <div class="stat-card"><span class="stat-val">3</span><div class="stat-label">Metrics</div></div>
      <div class="stat-card"><span class="stat-val">2</span><div class="stat-label">Encodings</div></div>
      <div class="stat-card"><span class="stat-val">∞</span><div class="stat-label">Datasets</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1rem;font-weight:600;color:#8888aa;"
                "margin-bottom:0.8rem'>How it works</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="steps">
      <div class="step-card">
        <div class="step-num">Step 1</div>
        <span class="step-icon">📂</span>
        <div class="step-title">Upload Data</div>
        <div class="step-desc">Load a CSV, inspect stats, handle missing values, encode &amp; scale features.</div>
      </div>
      <div class="step-card">
        <div class="step-num">Step 2</div>
        <span class="step-icon">🏋️</span>
        <div class="step-title">Train Models</div>
        <div class="step-desc">Pick from LR, RF, XGBoost, LightGBM. Compare accuracy, F1, and ROC-AUC.</div>
      </div>
      <div class="step-card">
        <div class="step-num">Step 3</div>
        <span class="step-icon">🔮</span>
        <div class="step-title">Predict</div>
        <div class="step-desc">Enter feature values and get a live prediction with class probabilities.</div>
      </div>
      <div class="step-card">
        <div class="step-num">Step 4</div>
        <span class="step-icon">🔍</span>
        <div class="step-title">Explain</div>
        <div class="step-desc">Explore global SHAP importance and per-sample waterfall explanations.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;color:#3a3a5a;text-align:center'>"
                "Tip — use the bundled <code>data/iris.csv</code> to try it out instantly.</div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# UPLOAD DATA
# ══════════════════════════════════════════════════════════════════════════
elif page == "📂  Upload Data":
    page_header("Upload & Explore", "Load a CSV, inspect it, then preprocess for training.")

    uploaded = st.file_uploader("Drop a CSV file here or click to browse", type=["csv"])

    if uploaded:
        with st.spinner("Reading file…"):
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.stop()
            st.session_state.df = df

        # Quick stats row
        n_rows, n_cols = df.shape
        n_missing = int(df.isnull().sum().sum())
        n_num = int(df.select_dtypes(include=np.number).shape[1])
        n_cat = n_cols - n_num

        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card"><span class="stat-val">{n_rows:,}</span><div class="stat-label">Rows</div></div>
          <div class="stat-card"><span class="stat-val">{n_cols}</span><div class="stat-label">Columns</div></div>
          <div class="stat-card"><span class="stat-val">{n_num}</span><div class="stat-label">Numeric</div></div>
          <div class="stat-card"><span class="stat-val">{n_cat}</span><div class="stat-label">Categorical</div></div>
          <div class="stat-card"><span class="stat-val">{n_missing}</span><div class="stat-label">Missing cells</div></div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋  Preview", "📊  Statistics", "🕳️  Missing Values", "🔎  Feature Summary"])

        with tab1:
            st.dataframe(df.head(20), use_container_width=True, height=320)

        with tab2:
            st.dataframe(df.describe(include="all").T, use_container_width=True)

        with tab3:
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if missing.empty:
                st.markdown('<div class="success-box">✦ No missing values — dataset is clean.</div>',
                            unsafe_allow_html=True)
            else:
                miss_df = missing.rename("Missing Count").to_frame()
                miss_df["Missing %"] = (missing / len(df) * 100).round(2)
                st.dataframe(miss_df, use_container_width=True)
                st.markdown('<div class="info-box">Missing values will be imputed automatically '
                            '(median for numeric, mode for categorical).</div>',
                            unsafe_allow_html=True)

        with tab4:
            st.dataframe(get_feature_summary(df), use_container_width=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Preprocessing")

        valid_targets = [c for c in df.columns if df[c].nunique() < 20]
        if not valid_targets:
            st.markdown('<div class="error-box">❌ No valid target column found '
                        '(all columns have ≥ 20 unique values).</div>',
                        unsafe_allow_html=True)
            st.stop()

        col_left, col_right = st.columns([1, 1])
        with col_left:
            target_col = st.selectbox(
                "Target column (low-cardinality only)",
                valid_targets,
                help="Only columns with < 20 unique values are shown.",
            )
        with col_right:
            encoding_strategy = st.radio(
                "Encoding strategy",
                ["Label Encoding", "One-Hot Encoding"],
                horizontal=True,
            )

        # Class distribution
        class_counts = df[target_col].value_counts()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("**Class distribution**")
            st.dataframe(class_counts.rename("Count").to_frame(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            if class_counts.min() < 2:
                st.markdown(
                    '<div class="warn-box">⚠️ Some classes have fewer than 2 samples. '
                    'Stratified splitting will be skipped.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="success-box">✦ {len(class_counts)} classes detected — '
                    f'all have ≥ 2 samples. Stratified split enabled.</div>',
                    unsafe_allow_html=True)

        st.markdown("")
        if st.button("🚀  Process Dataset", type="primary"):
            with st.spinner("Encoding and scaling…"):
                try:
                    df_clean = handle_missing_values(df.copy())
                    X_enc, y, encoders, cat_cols, num_cols = encode_features(
                        df_clean, target_col, strategy=encoding_strategy)
                    X_scaled, scaler = scale_features(X_enc)
                    feature_names = list(X_enc.columns)

                    st.session_state.df_processed      = df_clean
                    st.session_state.X                 = X_scaled
                    st.session_state.y                 = y
                    st.session_state.encoders          = encoders
                    st.session_state.scaler            = scaler
                    st.session_state.feature_names     = feature_names
                    st.session_state.cat_cols          = cat_cols
                    st.session_state.num_cols          = num_cols
                    st.session_state.target_col        = target_col
                    st.session_state.encoding_strategy = encoding_strategy
                    st.session_state.models            = None
                    st.session_state.results           = None
                    st.session_state.shap_values       = None

                    st.success(f"✅ Done! {X_scaled.shape[1]} features ready.")
                    st.markdown(
                        f'<div class="info-box">'
                        f'Encoding: <b>{encoding_strategy}</b> &nbsp;·&nbsp; '
                        f'Categorical: <b>{len(cat_cols)}</b> &nbsp;·&nbsp; '
                        f'Numeric: <b>{len(num_cols)}</b> &nbsp;·&nbsp; '
                        f'Final features: <b>{X_scaled.shape[1]}</b>'
                        f'</div>',
                        unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Processing failed: {e}</div>',
                                unsafe_allow_html=True)

    elif st.session_state.df is not None:
        st.markdown('<div class="info-box">A dataset is already loaded. '
                    'Upload a new file to replace it.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">👆 Upload a CSV file to get started.</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════
elif page == "🏋️  Train Model":
    page_header("Train & Evaluate", "Select models, configure the split, and compare results.")

    if st.session_state.X is None:
        st.markdown('<div class="warn-box">⚠️ Please upload and process a dataset first.</div>',
                    unsafe_allow_html=True)
        st.stop()

    X = st.session_state.X
    y = st.session_state.y

    # Config row
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    cfg1, cfg2, cfg3 = st.columns([2, 2, 3])
    with cfg1:
        test_size = st.slider("Test split (%)", 10, 40, 20) / 100
    with cfg2:
        random_state = st.number_input("Random seed", 0, 999, 42)
    with cfg3:
        model_options = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
        selected_models = st.multiselect(
            "Models to train", model_options, default=model_options)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    if st.button("🏋️  Train Selected Models", type="primary"):
        if not selected_models:
            st.markdown('<div class="error-box">Select at least one model.</div>',
                        unsafe_allow_html=True)
            st.stop()

        with st.spinner("Training — this may take a moment…"):
            try:
                X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
                models, results = train_all_models(
                    X_train, X_test, y_train, y_test, selected_models, random_state)

                st.session_state.models      = models
                st.session_state.results     = results
                st.session_state.X_test_arr  = X_test
                st.session_state.X_train_arr = X_train
                st.session_state.y_test      = y_test
                st.session_state.shap_values = None

                models_dir = os.path.join(os.path.dirname(__file__), "models")
                os.makedirs(models_dir, exist_ok=True)
                for name, mdl in models.items():
                    joblib.dump(mdl, os.path.join(models_dir,
                                f"{name.replace(' ', '_')}.joblib"))
                joblib.dump(st.session_state.scaler,
                            os.path.join(models_dir, "scaler.joblib"))
                joblib.dump(st.session_state.encoders,
                            os.path.join(models_dir, "encoders.joblib"))
                joblib.dump(st.session_state.feature_names,
                            os.path.join(models_dir, "feature_names.joblib"))

                st.success("✅ Training complete — models saved to `models/`.")
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Training failed: {e}</div>',
                            unsafe_allow_html=True)
                st.stop()

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.results:
        results = st.session_state.results

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### 📊 Results Summary")

        # Metrics table
        rows = []
        for name, r in results.items():
            rows.append({
                "Model":      name,
                "Accuracy":   f"{r['accuracy']:.4f}",
                "F1 (macro)": f"{r['f1']:.4f}",
                "ROC-AUC":    f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        st.pyplot(plot_model_comparison(results), use_container_width=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### 🔬 Per-Model Details")

        model_tabs = st.tabs(list(results.keys()))
        for tab, (name, r) in zip(model_tabs, results.items()):
            with tab:
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy",   f"{r['accuracy']:.4f}")
                m2.metric("F1 (macro)", f"{r['f1']:.4f}")
                m3.metric("ROC-AUC",    f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A")

                col_cm, col_roc = st.columns(2)
                with col_cm:
                    st.pyplot(plot_confusion_matrix(r["y_test"], r["y_pred"]),
                              use_container_width=True)
                with col_roc:
                    if r["y_prob"] is not None:
                        st.pyplot(plot_roc_curve(r["y_test"], r["y_prob"], name),
                                  use_container_width=True)
                    else:
                        st.info("ROC curve not available.")

                with st.expander("📄 Classification Report"):
                    st.code(r["report"], language=None)

# ══════════════════════════════════════════════════════════════════════════
# PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predictions":
    page_header("Make a Prediction", "Enter feature values and get an instant prediction.")

    if st.session_state.models is None:
        st.markdown('<div class="warn-box">⚠️ Please train models first.</div>',
                    unsafe_allow_html=True)
        st.stop()

    models        = st.session_state.models
    encoders      = st.session_state.encoders
    scaler        = st.session_state.scaler
    feature_names = st.session_state.feature_names
    cat_cols      = st.session_state.cat_cols
    df_proc       = st.session_state.df_processed
    target_col    = st.session_state.target_col
    strategy      = st.session_state.encoding_strategy

    selected_model_name = st.selectbox(
        "Model", list(models.keys()),
        help="Choose which trained model to use for prediction.")
    model = models[selected_model_name]

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Feature Inputs")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    input_values = {}
    original_feature_cols = [c for c in df_proc.columns if c != target_col]
    cols = st.columns(3)

    for i, feat in enumerate(original_feature_cols):
        with cols[i % 3]:
            if feat in cat_cols:
                le = encoders.get(feat)
                options = list(le.classes_) if le is not None else \
                          df_proc[feat].dropna().unique().tolist()
                input_values[feat] = st.selectbox(feat, options, key=f"pred_{feat}")
            else:
                col_min  = float(df_proc[feat].min())
                col_max  = float(df_proc[feat].max())
                col_mean = float(df_proc[feat].mean())
                input_values[feat] = st.number_input(
                    feat, min_value=col_min, max_value=col_max,
                    value=col_mean, key=f"pred_{feat}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")

    if st.button("🔮  Predict", type="primary"):
        try:
            row = pd.DataFrame([input_values])

            if strategy == "Label Encoding":
                for col in cat_cols:
                    le = encoders.get(col)
                    if le:
                        row[col] = le.transform(row[col].astype(str))
            else:
                for col in cat_cols:
                    le = encoders.get(col)
                    if le:
                        row[col] = le.transform(row[col].astype(str))
                row = pd.get_dummies(row, columns=cat_cols, drop_first=True)
                for c in feature_names:
                    if c not in row.columns:
                        row[c] = 0
                row = row[feature_names]

            row_scaled  = scaler.transform(row)
            pred_class  = model.predict(row_scaled)[0]
            pred_proba  = (model.predict_proba(row_scaled)[0]
                           if hasattr(model, "predict_proba") else None)

            le_target = encoders.get("target")
            try:
                pred_label = str(le_target.inverse_transform([pred_class])[0]) \
                             if le_target else str(pred_class)
            except Exception:
                pred_label = str(pred_class)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### 🎯 Result")

            if pred_proba is not None:
                confidence = pred_proba.max() * 100
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="result-label">Predicted Class</div>'
                        f'<div class="result-value">{pred_label}</div>'
                        f'<div class="result-sub">using {selected_model_name}</div>'
                        f'</div>', unsafe_allow_html=True)
                with r2:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="result-label">Confidence</div>'
                        f'<div class="result-value">{confidence:.1f}%</div>'
                        f'<div class="result-sub">max class probability</div>'
                        f'</div>', unsafe_allow_html=True)
                with r3:
                    n_classes = len(pred_proba)
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="result-label">Classes</div>'
                        f'<div class="result-value">{n_classes}</div>'
                        f'<div class="result-sub">in this dataset</div>'
                        f'</div>', unsafe_allow_html=True)

                # Probability chart
                st.markdown("")
                if le_target is not None:
                    try:
                        class_labels = [str(c) for c in le_target.classes_]
                    except Exception:
                        class_labels = [str(i) for i in range(len(pred_proba))]
                else:
                    class_labels = [str(i) for i in range(len(pred_proba))]

                fig, ax = plt.subplots(figsize=(7, max(2, len(class_labels) * 0.55)))
                colors = ["#7efff5" if p == pred_proba.max() else "#a78bfa"
                          for p in pred_proba]
                bars = ax.barh(class_labels, pred_proba, color=colors, alpha=0.85,
                               height=0.55)
                for bar, prob in zip(bars, pred_proba):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{prob:.3f}", va="center", color="#e0e0f0", fontsize=9)
                ax.set_xlim(0, 1.15)
                ax.set_xlabel("Probability", color="#6b6b8a")
                ax.set_title("Class Probabilities", color="#e0e0f0", fontsize=11)
                fig.patch.set_facecolor("#0d0d1a")
                ax.set_facecolor("#12122a")
                ax.tick_params(colors="#6b6b8a")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#ffffff10")
                ax.grid(axis="x", color="#ffffff08", linestyle="--", linewidth=0.5)
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            else:
                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="result-label">Predicted Class</div>'
                    f'<div class="result-value">{pred_label}</div>'
                    f'</div>', unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Prediction failed: {e}</div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# EXPLAIN MODEL
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍  Explain Model":
    page_header("Explain Model Decisions",
                "Global feature importance and per-sample SHAP waterfall charts.")

    if st.session_state.models is None:
        st.markdown('<div class="warn-box">⚠️ Please train models first.</div>',
                    unsafe_allow_html=True)
        st.stop()

    models        = st.session_state.models
    feature_names = st.session_state.feature_names
    X_test        = st.session_state.X_test_arr

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    e1, e2 = st.columns([2, 2])
    with e1:
        selected_model_name = st.selectbox("Model to explain", list(models.keys()))
    with e2:
        max_samples = min(300, len(X_test))
        n_samples = st.slider(
            "SHAP sample size (fewer = faster)",
            min_value=20, max_value=max_samples,
            value=min(100, max_samples), step=10,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    model = models[selected_model_name]
    st.markdown("")

    if st.button("⚡  Compute SHAP Values", type="primary"):
        with st.spinner("Computing SHAP values…"):
            try:
                X_sample = X_test[:n_samples]
                shap_vals, _ = compute_shap_values(model, X_sample, selected_model_name)
                st.session_state.shap_values        = shap_vals
                st.session_state["shap_X_sample"]   = X_sample
                st.session_state["shap_model_name"] = selected_model_name
                st.success("✅ SHAP values computed.")
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ SHAP failed: {e}</div>',
                            unsafe_allow_html=True)
                st.stop()

    if st.session_state.shap_values is not None:
        shap_vals = st.session_state.shap_values
        X_sample  = st.session_state.get("shap_X_sample", X_test[:n_samples])

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        tab_global, tab_local = st.tabs(["🌍  Global Explanations", "🔬  Local Explanation"])

        with tab_global:
            st.markdown("#### Feature Importance")
            st.markdown(
                '<div class="info-box">Mean absolute SHAP value per feature — '
                'longer bar = stronger average influence on predictions.</div>',
                unsafe_allow_html=True)
            st.pyplot(plot_feature_importance(shap_vals, feature_names),
                      use_container_width=True)

            st.markdown("#### SHAP Summary Plot")
            st.markdown(
                '<div class="info-box">Each dot = one sample. '
                'Color = feature value (red high / blue low). '
                'X-axis = SHAP impact on model output.</div>',
                unsafe_allow_html=True)
            st.pyplot(plot_shap_summary(shap_vals, X_sample, feature_names),
                      use_container_width=True)

        with tab_local:
            st.markdown("#### Per-Sample Waterfall")
            sample_idx = st.slider("Sample index", 0, len(shap_vals) - 1, 0)

            st.pyplot(plot_shap_waterfall(shap_vals, sample_idx, feature_names),
                      use_container_width=True)

            # Plain-language card
            mdl_name  = st.session_state.get("shap_model_name", selected_model_name)
            model_obj = models[mdl_name]
            pred_cls  = model_obj.predict(X_sample[sample_idx:sample_idx + 1])[0]
            le_target = st.session_state.encoders.get("target")
            try:
                pred_label = str(le_target.inverse_transform([pred_cls])[0]) \
                             if le_target else str(pred_cls)
            except Exception:
                pred_label = str(pred_cls)

            explanation = generate_plain_language_explanation(
                shap_vals, sample_idx, feature_names, pred_label)
            st.markdown(
                f'<div class="info-box">💬 {explanation}</div>',
                unsafe_allow_html=True)
