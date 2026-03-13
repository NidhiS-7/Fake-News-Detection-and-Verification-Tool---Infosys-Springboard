from pathlib import Path

import pandas as pd
import streamlit as st

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import get_training_metrics, load_dataset, load_metrics, train_model  # noqa: E402

DATASET_PATH = Path("data/dataset.csv")

st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("Admin Dashboard")

col1, col2, col3 = st.columns(3)
metrics = load_metrics()
training = get_training_metrics()

dataset_rows = 0
if DATASET_PATH.exists():
    try:
        dataset_rows = len(load_dataset(DATASET_PATH))
    except Exception:
        dataset_rows = 0

with col1:
    st.metric("Articles Analyzed", metrics.get("articles_analyzed", 0))
with col2:
    st.metric("Dataset Rows", dataset_rows)
with col3:
    st.metric("Transformer Enabled", str(training.get("transformer_enabled", False)))

if st.button("Retrain Model"):
    with st.spinner("Training model..."):
        train_model(DATASET_PATH)
    st.success("Model retrained.")
    st.rerun()

st.subheader("Evaluation Metrics")
st.json(training)

st.subheader("Dataset Preview")
if DATASET_PATH.exists():
    df = pd.read_csv(DATASET_PATH)
    st.dataframe(df.head(50), use_container_width=True)
else:
    st.warning("Dataset not found.")
