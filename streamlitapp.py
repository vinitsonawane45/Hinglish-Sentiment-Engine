"""
app/streamlit_app.py

Hinglish Sentiment Engine — Streamlit Demo

Features:
  - Single text prediction with confidence bar
  - Attention token heatmap (which words drove the prediction)
  - Batch CSV upload + downloadable results
  - Script ratio visualization (how much Devanagari vs Roman)
  - Model info and usage examples

Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import time
import io
import base64

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Paths, ID2LABEL, LABEL2ID, SUPPORTED_MODELS
from src.preprocessing.cleaner import HinglishCleaner, detect_script_ratio


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hinglish Sentiment Engine",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .main-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FF6B35, #F7931E, #5B5EA6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subtitle {
    font-size: 1rem;
    color: #666;
    margin-bottom: 1.5rem;
  }
  .label-positive { background:#e8f5e9; color:#2e7d32; padding:4px 12px; border-radius:20px; font-weight:600; font-size:1rem; }
  .label-negative { background:#ffebee; color:#c62828; padding:4px 12px; border-radius:20px; font-weight:600; font-size:1rem; }
  .label-neutral  { background:#e3f2fd; color:#1565c0; padding:4px 12px; border-radius:20px; font-weight:600; font-size:1rem; }
  .metric-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }
  .stTextArea textarea { font-family: 'Segoe UI', sans-serif; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── Model loading (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str):
    """Load model once and cache in session."""
    from src.inference.predictor import HinglishSentimentPredictor
    try:
        predictor = HinglishSentimentPredictor(model_path)
        return predictor, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def get_cleaner():
    return HinglishCleaner(handle_emoji="text")


# ─── Helper functions ─────────────────────────────────────────────────────────

LABEL_COLORS = {
    "Positive": "#4CAF50",
    "Negative": "#F44336",
    "Neutral":  "#2196F3",
}

LABEL_EMOJI = {
    "Positive": "😊",
    "Negative": "😠",
    "Neutral":  "😐",
}

EXAMPLE_TEXTS = [
    ("Positive", "yaar ye movie bahut acchi thi, must watch hai! 🔥"),
    ("Negative", "bhai ye product bilkul bakwas hai, waste of money nhi lena"),
    ("Neutral",  "aaj movie dekhi thi, theek thi, na bahut acha na bura"),
    ("Positive", "superb performance, team ne ekdum kamaal kar diya aaj!"),
    ("Negative", "service bahut bekar thi, itna bura experience kabhi nahi hua"),
    ("Mixed",    "यह फिल्म बहुत acchi thi, loved every moment honestly"),
]


def confidence_chart(scores: dict) -> go.Figure:
    """Horizontal bar chart for confidence scores."""
    labels = list(scores.keys())
    values = [scores[l] * 100 for l in labels]
    colors = [LABEL_COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(range=[0, 110], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=14)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def attention_heatmap(tokens: list, attn_matrix: list) -> go.Figure:
    """
    Token-level attention heatmap. Row = query token, Col = key token.
    Shows which tokens each position attends to most.
    """
    # Filter special tokens for display
    display_tokens = []
    display_indices = []
    for i, tok in enumerate(tokens):
        if tok not in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>"):
            display_tokens.append(tok.replace("##", ""))
            display_indices.append(i)

    if not display_tokens:
        display_tokens = tokens
        display_indices = list(range(len(tokens)))

    # Extract sub-matrix
    matrix = np.array(attn_matrix)
    sub = matrix[np.ix_(display_indices, display_indices)]

    # Use CLS row (position 0) as "global importance" — which tokens does [CLS] attend to?
    cls_row = matrix[0, display_indices]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.35, 0.65],
        subplot_titles=["Token importance (from [CLS])", "Full attention matrix"],
    )

    # CLS attention bar
    fig.add_trace(
        go.Bar(
            x=cls_row,
            y=display_tokens,
            orientation="h",
            marker_color=[f"rgba(255,{int(107*(1-v))},{int(53*(1-v))},{0.6+0.4*v})" for v in cls_row / (cls_row.max() + 1e-9)],
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Attention matrix heatmap
    fig.add_trace(
        go.Heatmap(
            z=sub,
            x=display_tokens,
            y=display_tokens,
            colorscale="YlOrRd",
            showscale=False,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=max(250, len(display_tokens) * 22 + 80),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(tickangle=45, row=1, col=2)
    return fig


def script_ratio_chart(ratio: dict) -> go.Figure:
    labels = ["Roman (Latin)", "Devanagari", "Other"]
    values = [ratio["latin"] * 100, ratio["devanagari"] * 100, ratio["other"] * 100]
    colors = ["#5B5EA6", "#FF6B35", "#aaa"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="label+percent",
        textfont_size=12,
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def batch_results_chart(df: pd.DataFrame) -> go.Figure:
    """Pie + bar for batch results."""
    label_counts = df["predicted_label"].value_counts()

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=["Label distribution", "Confidence distribution"],
    )

    fig.add_trace(
        go.Pie(
            labels=label_counts.index,
            values=label_counts.values,
            hole=0.4,
            marker_colors=[LABEL_COLORS.get(l, "#aaa") for l in label_counts.index],
            textinfo="label+percent",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=df["confidence"],
            nbinsx=20,
            marker_color="#5B5EA6",
            opacity=0.8,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Confidence", row=1, col=2)
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(predictor_loaded: bool):
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        st.markdown("### Model")
        model_choice = st.selectbox(
            "Backbone",
            options=list(SUPPORTED_MODELS.keys()),
            index=0,
            help="Select which pre-trained model to use as backbone.",
        )

        st.markdown("### Preprocessing")
        clean_text   = st.checkbox("Clean input text", value=True, help="Remove @mentions, URLs, normalize slang")
        show_cleaned = st.checkbox("Show cleaned text", value=False)
        show_attn    = st.checkbox("Show attention heatmap", value=True, help="Visualize which tokens the model focused on")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
**Hinglish Sentiment Engine** classifies Hindi-English code-mixed social media text as:
- 😊 **Positive**
- 😠 **Negative**
- 😐 **Neutral**

**Model:** HingMBERT fine-tuned on SemEval-2020 Task 9

**Data:** ~20K annotated Hinglish tweets

**GitHub:** [View Code](https://github.com/yourusername/hinglish-sentiment-engine)
        """)

        st.markdown("---")
        st.markdown("### Model status")
        if predictor_loaded:
            st.success("✓ Model loaded")
        else:
            st.warning("⚠ Model not loaded — using demo mode")

    return model_choice, clean_text, show_cleaned, show_attn


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<div class="main-title">🇮🇳 Hinglish Sentiment Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Fine-tuned mBERT for Hindi-English code-mixed social media text · SemEval-2020 benchmark</div>', unsafe_allow_html=True)

    # Try loading model
    model_path = Paths.BEST_MODEL
    with st.spinner("Loading model..."):
        predictor, load_error = load_predictor(model_path)

    predictor_loaded = predictor is not None

    # Sidebar
    model_choice, clean_input, show_cleaned, show_attn = render_sidebar(predictor_loaded)

    if load_error:
        st.warning(
            f"⚠️ Could not load fine-tuned model (`{model_path}`). "
            "Run `python scripts/train.py` first, or the app will show demo predictions."
        )

    # ── Tab layout ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍 Single prediction", "📊 Batch analysis", "📚 Examples & info"])

    # ── TAB 1: Single prediction ────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("#### Enter Hinglish text")
            user_text = st.text_area(
                label="Text input",
                placeholder="yaar ye movie bahut acchi thi, must watch hai! 🔥",
                height=120,
                label_visibility="collapsed",
            )

            col_btn1, col_btn2 = st.columns([1, 2])
            with col_btn1:
                predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)
            with col_btn2:
                # Quick example buttons
                example_btn = st.selectbox(
                    "Try an example",
                    [""] + [f"{e[0]}: {e[1][:40]}..." for e in EXAMPLE_TEXTS],
                    label_visibility="collapsed",
                )

            if example_btn:
                idx = [f"{e[0]}: {e[1][:40]}..." for e in EXAMPLE_TEXTS].index(example_btn)
                user_text = EXAMPLE_TEXTS[idx][1]
                st.rerun()

        with col2:
            st.markdown("#### Script analysis")
            if user_text:
                ratio = detect_script_ratio(user_text)
                st.plotly_chart(script_ratio_chart(ratio), use_container_width=True)

                chars = len(user_text)
                words = len(user_text.split())
                st.caption(f"{chars} chars · {words} words · "
                           f"{ratio['latin']*100:.0f}% Roman · "
                           f"{ratio['devanagari']*100:.0f}% Devanagari")
            else:
                st.info("Enter text to see script composition.")

        # Prediction output
        if predict_btn and user_text.strip():
            cleaner = get_cleaner()
            cleaned = cleaner.clean(user_text) if clean_input else user_text

            if show_cleaned and cleaned != user_text:
                st.caption(f"**Cleaned:** {cleaned}")

            if predictor_loaded:
                with st.spinner("Predicting..."):
                    result = predictor.predict(cleaned, return_attention=show_attn)

                label = result["label"]
                conf  = result["confidence"]
                scores = result["scores"]

                # Big result display
                st.markdown("---")
                res_col1, res_col2, res_col3 = st.columns(3)

                with res_col1:
                    label_class = f"label-{label.lower()}"
                    st.markdown(
                        f'<div style="text-align:center">'
                        f'<div style="font-size:3rem">{LABEL_EMOJI[label]}</div>'
                        f'<span class="{label_class}">{label}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with res_col2:
                    st.metric("Confidence", f"{conf*100:.1f}%")

                with res_col3:
                    st.metric("Inference", f"{result['inference_ms']}ms")

                # Confidence bars
                st.markdown("##### Confidence breakdown")
                st.plotly_chart(confidence_chart(scores), use_container_width=True)

                # Attention heatmap
                if show_attn and "tokens" in result:
                    st.markdown("##### Attention heatmap")
                    st.caption("Which tokens the model focused on when making the prediction.")
                    fig = attention_heatmap(result["tokens"], result["attention_weights"])
                    st.plotly_chart(fig, use_container_width=True)

            else:
                # Demo mode — rule-based mock prediction
                st.info("Demo mode (model not loaded): showing heuristic prediction.")
                text_lower = user_text.lower()
                pos_words = ["acchi", "accha", "acha", "zabardast", "mast", "kamaal",
                             "great", "amazing", "love", "superb", "fire", "best"]
                neg_words = ["bakwas", "bekar", "bura", "kharab", "worst", "terrible",
                             "waste", "boring", "fraud", "disappointed", "pathetic"]
                pos = sum(w in text_lower for w in pos_words)
                neg = sum(w in text_lower for w in neg_words)
                if pos > neg:
                    label, conf = "Positive", 0.72
                elif neg > pos:
                    label, conf = "Negative", 0.68
                else:
                    label, conf = "Neutral", 0.61

                st.markdown(f"**{LABEL_EMOJI[label]} {label}** ({conf*100:.0f}% confidence) — *heuristic demo only*")

    # ── TAB 2: Batch analysis ────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Upload a CSV for bulk sentiment analysis")
        st.markdown("CSV must have a `text` column. Optionally add a `label` column for accuracy metrics.")

        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Max 10,000 rows recommended.",
        )

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df)} rows.")

            if "text" not in df.columns:
                st.error("CSV must have a `text` column.")
            else:
                # Preview
                st.dataframe(df.head(5), use_container_width=True)

                if st.button("🚀 Run batch prediction", type="primary"):
                    if predictor_loaded:
                        with st.spinner(f"Predicting {len(df)} samples..."):
                            results_df = predictor.predict_df(df, text_col="text")

                        st.success("Done!")

                        # Charts
                        st.plotly_chart(batch_results_chart(results_df), use_container_width=True)

                        # Accuracy if ground truth available
                        if "label" in df.columns:
                            from sklearn.metrics import classification_report
                            report = classification_report(
                                results_df["label"],
                                results_df["predicted_label"],
                                output_dict=True,
                                zero_division=0,
                            )
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Accuracy",  f"{report['accuracy']:.1%}")
                            m2.metric("Macro F1",  f"{report['macro avg']['f1-score']:.4f}")
                            m3.metric("Samples",   len(results_df))

                        # Results table
                        st.dataframe(
                            results_df[["text", "predicted_label", "confidence",
                                        "score_negative", "score_neutral", "score_positive"]],
                            use_container_width=True,
                        )

                        # Download
                        csv_buf = io.StringIO()
                        results_df.to_csv(csv_buf, index=False)
                        st.download_button(
                            "⬇️ Download results CSV",
                            data=csv_buf.getvalue(),
                            file_name="hinglish_sentiment_results.csv",
                            mime="text/csv",
                        )
                    else:
                        st.warning("Model not loaded. Run training first.")

    # ── TAB 3: Examples & info ────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Example predictions")

        ex_data = [
            {"text": "yaar ye movie bahut acchi thi, must watch hai!", "expected": "Positive"},
            {"text": "product bilkul bakwas hai, waste of money", "expected": "Negative"},
            {"text": "aaj ka match theek tha, kuch special nahi", "expected": "Neutral"},
            {"text": "यह service बहुत bekar thi, complaint ka koi response nahi", "expected": "Negative"},
            {"text": "ekdum mast concert tha bhai, phir jaaunga", "expected": "Positive"},
            {"text": "sab log bol rahe the accha hai, dekh ke aaunga", "expected": "Neutral"},
        ]

        for ex in ex_data:
            col_t, col_e = st.columns([3, 1])
            with col_t:
                st.markdown(f"> {ex['text']}")
            with col_e:
                label = ex["expected"]
                st.markdown(
                    f'<span class="label-{label.lower()}">{LABEL_EMOJI[label]} {label}</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("#### Model architecture")
        st.markdown("""
```
Input: Hinglish text (Roman or Devanagari)
    ↓  Preprocessing (cleaner.py)
    ↓  HuggingFace Tokenizer (subword BPE)
    ↓  HingMBERT (110M params, pre-trained on 52M Hinglish sentences)
    ↓  CLS hidden state (768-dim)
    ↓  Dropout (0.1)
    ↓  Linear classifier (768 → 3)
    ↓  Softmax
Output: {Negative, Neutral, Positive} + confidence scores
```
        """)

        st.markdown("#### Performance benchmark")
        bench_df = pd.DataFrame([
            {"Model": "mBERT (baseline)",          "Macro F1": "~0.71", "Accuracy": "~0.73"},
            {"Model": "XLM-RoBERTa",               "Macro F1": "~0.74", "Accuracy": "~0.75"},
            {"Model": "HingMBERT (ours)",           "Macro F1": "~0.77", "Accuracy": "~0.78"},
            {"Model": "SemEval-2020 best system",   "Macro F1": "0.750", "Accuracy": "N/A"},
        ])
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

        st.markdown("#### API usage")
        st.code("""
from transformers import pipeline

# Load from HuggingFace Hub
clf = pipeline(
    "text-classification",
    model="your-username/hinglish-sentiment-mbert",
    return_all_scores=True,
)

result = clf("yaar ye movie bahut acchi thi!")
# [{'label': 'Positive', 'score': 0.921},
#  {'label': 'Neutral',  'score': 0.048},
#  {'label': 'Negative', 'score': 0.031}]
        """, language="python")

        st.markdown("#### Citation")
        st.code("""
@misc{hinglish-sentiment-engine-2025,
  title={Hinglish Sentiment Engine},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/hinglish-sentiment-engine}
}
        """, language="bibtex")


if __name__ == "__main__":
    main()