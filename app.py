import json
from pathlib import Path

import streamlit as st

from explainability import find_suspicious_claims, generate_explanation
from model import get_top_keywords, get_training_metrics, load_metrics, predict_news, save_metrics, train_model
from source_manager import is_source_trusted

METRICS_PATH = Path("metrics.json")


def initialize_metrics() -> None:
    if not METRICS_PATH.exists():
        save_metrics({"articles_analyzed": 0, "training": {}})


initialize_metrics()

st.set_page_config(page_title="Fake News Explainability Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #05101e 0%, #0a1628 100%); color: white;}
    .main-title {font-size: 40px; font-weight: 800; color: white; margin-bottom: 4px;}
    .sub-title {color: #cbd5e1; margin-bottom: 24px;}
    .card {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); padding: 18px; border-radius: 16px; margin-bottom: 16px;}
    .pill-real {display:inline-block; background:#163a24; color:#86efac; padding:8px 14px; border-radius:999px; font-weight:700;}
    .pill-fake {display:inline-block; background:#421717; color:#fca5a5; padding:8px 14px; border-radius:999px; font-weight:700;}
    .pill-true {display:inline-block; background:#163a24; color:#86efac; padding:8px 14px; border-radius:999px; font-weight:700;}
    .pill-false {display:inline-block; background:#4a3411; color:#fde68a; padding:8px 14px; border-radius:999px; font-weight:700;}
    .metric-label {color:#94a3b8; font-size:14px; margin-bottom:8px;}
    .metric-value {font-size:24px; font-weight:800; color:white;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Fake News Explainability Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid model: TF-IDF word features + character features + transformer embeddings + source trust + explainability</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Model Controls")
    if st.button("Retrain Model"):
        with st.spinner("Training model..."):
            train_model()
        st.success("Model retrained successfully.")
    training_metrics = get_training_metrics()
    if training_metrics:
        st.write("Latest Training Accuracy:", training_metrics.get("accuracy", "N/A"))
        st.write("Latest Training F1:", training_metrics.get("f1", "N/A"))
        st.write("Transformer Enabled:", training_metrics.get("transformer_enabled", False))

col_input, col_output = st.columns([1, 1])

with col_input:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    title = st.text_input("News Title", placeholder="Example: ISRO launches new Earth observation satellite")
    text = st.text_area("News Article", height=220, placeholder="Paste any real-world news article or headline here...")
    source = st.text_input("News Source", placeholder="Example: Reuters, BBC, NDTV")
    analyze = st.button("Analyze Article", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_output:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Expected Output Fields")
    st.json(
        {
            "prediction": {"prediction": "real/fake", "confidence": 0.0, "probabilities": {"real": 0.0, "fake": 0.0}},
            "suspicious_claims": ["sample phrase"],
            "explanation": ["sample explanation"],
            "source_trusted": True,
            "system_metrics": {"articles_analyzed": 0},
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)

if analyze:
    if not title.strip() and not text.strip():
        st.warning("Enter at least a title or article text.")
    else:
        with st.spinner("Analyzing article..."):
            prediction_data = predict_news(title=title, text=text, source=source)
            suspicious = find_suspicious_claims(f"{title} {text}")
            keywords = [word for word, _ in get_top_keywords(title=title, text=text, source=source, top_n=8)]
            trusted = is_source_trusted(source)

            explanation = generate_explanation(
            title=title,
            text=text,
            source=source,
            prediction=prediction_data["prediction"],
            confidence=prediction_data["confidence"],
            important_terms=keywords,
        )
            trusted = is_source_trusted(source)

            metrics = load_metrics()
            metrics["articles_analyzed"] = int(metrics.get("articles_analyzed", 0)) + 1
            save_metrics(metrics)

        pred_class = "pill-fake" if prediction_data["prediction"] == "fake" else "pill-real"
        pred_text = prediction_data["prediction"].upper()
        source_class = "pill-true" if trusted else "pill-false"
        source_text = "TRUE" if trusted else "FALSE"

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Prediction")
            st.markdown(f'<span class="{pred_class}">{pred_text}</span>', unsafe_allow_html=True)
            st.write(f"Confidence: **{prediction_data['confidence']:.4f}**")
            st.json(prediction_data)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Suspicious Claims")
            st.json(suspicious if suspicious else ["No suspicious claims found"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Source Trusted")
            st.markdown(f'<span class="{source_class}">{source_text}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Explanation")
            st.json(explanation)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("System Metrics")
            st.json(load_metrics())
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Top Important Terms")
            st.json(keywords)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
training_metrics = get_training_metrics()
if training_metrics:
    m1, m2, m3, m4 = st.columns(4)
    values = [
        ("Accuracy", training_metrics.get("accuracy", "N/A")),
        ("Precision", training_metrics.get("precision", "N/A")),
        ("Recall", training_metrics.get("recall", "N/A")),
        ("F1", training_metrics.get("f1", "N/A")),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4], values):
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
