import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from utils import build_combined_text, clean_text

MODEL_PATH = Path("fake_news_hybrid_model.pkl")
METRICS_PATH = Path("metrics.json")
DATASET_PATH = Path("data/dataset.csv")
TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(TRANSFORMER_MODEL_NAME)
    except Exception:
        return None


def normalize_label(value: str) -> str:
    value = str(value).strip().lower()
    mapping = {
        "true": "real",
        "real": "real",
        "1": "real",
        "reliable": "real",
        "false": "fake",
        "fake": "fake",
        "0": "fake",
        "unreliable": "fake",
    }
    return mapping.get(value, value)


REQUIRED_OUTPUT_COLUMNS = ["title", "text", "source", "label", "combined_text"]


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    lowered_cols = {col.lower(): col for col in df.columns}

    title_col = lowered_cols.get("title")
    text_col = lowered_cols.get("text") or lowered_cols.get("article") or lowered_cols.get("content")
    source_col = lowered_cols.get("source") or lowered_cols.get("news_source") or lowered_cols.get("domain")
    label_col = lowered_cols.get("label") or lowered_cols.get("target") or lowered_cols.get("class")

    if text_col is None or label_col is None:
        raise ValueError("Dataset must contain at least 'text' and 'label' columns.")

    out = pd.DataFrame()
    out["title"] = df[title_col].fillna("").astype(str) if title_col else ""
    out["text"] = df[text_col].fillna("").astype(str)
    out["source"] = df[source_col].fillna("").astype(str) if source_col else ""
    out["label"] = df[label_col].astype(str).map(normalize_label)

    out = out[out["label"].isin(["real", "fake"])].copy()
    out["title"] = out["title"].map(clean_text)
    out["text"] = out["text"].map(clean_text)
    out["source"] = out["source"].map(clean_text)
    out["combined_text"] = out.apply(
        lambda row: build_combined_text(row["title"], row["text"], row["source"]), axis=1
    )
    out = out[out["combined_text"].str.len() > 0].reset_index(drop=True)
    return out[REQUIRED_OUTPUT_COLUMNS]


def _balanced_class_weight(df: pd.DataFrame) -> str:
    return "balanced"


def _build_word_branch() -> Tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=30000,
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        C=2.0,
        solver="liblinear",
        random_state=42,
    )
    return vectorizer, clf


def _build_char_branch() -> Tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=25000,
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        C=1.5,
        solver="liblinear",
        random_state=42,
    )
    return vectorizer, clf


def _evaluate_from_probs(y_true: np.ndarray, probs_fake: np.ndarray) -> Dict:
    labels = np.where(probs_fake >= 0.5, "fake", "real")
    cm = confusion_matrix(y_true, labels, labels=["real", "fake"])
    report = classification_report(y_true, labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, labels)), 4),
        "precision": round(float(precision_score(y_true, labels, pos_label="fake", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, labels, pos_label="fake", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, labels, pos_label="fake", zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return metrics


def train_model(dataset_path: Path = DATASET_PATH) -> Dict:
    df = load_dataset(dataset_path)
    if len(df) < 10:
        raise ValueError("Dataset is too small. Add more rows before training.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["combined_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    word_vec, word_clf = _build_word_branch()
    char_vec, char_clf = _build_char_branch()

    Xw_train = word_vec.fit_transform(X_train)
    Xw_test = word_vec.transform(X_test)
    word_clf.fit(Xw_train, y_train)
    word_probs = word_clf.predict_proba(Xw_test)
    word_fake_probs = word_probs[:, list(word_clf.classes_).index("fake")]

    Xc_train = char_vec.fit_transform(X_train)
    Xc_test = char_vec.transform(X_test)
    char_clf.fit(Xc_train, y_train)
    char_probs = char_clf.predict_proba(Xc_test)
    char_fake_probs = char_probs[:, list(char_clf.classes_).index("fake")]

    embedder = _load_sentence_transformer()
    bert_clf = None
    bert_fake_probs = None
    transformer_enabled = False

    if embedder is not None:
        try:
            Xb_train = embedder.encode(list(X_train), batch_size=32, show_progress_bar=False)
            Xb_test = embedder.encode(list(X_test), batch_size=32, show_progress_bar=False)
            bert_clf = LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                C=2.0,
                solver="liblinear",
                random_state=42,
            )
            bert_clf.fit(Xb_train, y_train)
            bert_probs = bert_clf.predict_proba(Xb_test)
            bert_fake_probs = bert_probs[:, list(bert_clf.classes_).index("fake")]
            transformer_enabled = True
        except Exception:
            bert_clf = None
            bert_fake_probs = None
            transformer_enabled = False

    if transformer_enabled and bert_fake_probs is not None:
        ensemble_fake_probs = (0.40 * word_fake_probs) + (0.25 * char_fake_probs) + (0.35 * bert_fake_probs)
    else:
        ensemble_fake_probs = (0.62 * word_fake_probs) + (0.38 * char_fake_probs)

    evaluation = _evaluate_from_probs(y_test.to_numpy(), ensemble_fake_probs)
    evaluation["dataset_rows"] = int(len(df))
    evaluation["train_rows"] = int(len(X_train))
    evaluation["test_rows"] = int(len(X_test))
    evaluation["transformer_enabled"] = transformer_enabled

    artifact = {
        "word_vectorizer": word_vec,
        "word_clf": word_clf,
        "char_vectorizer": char_vec,
        "char_clf": char_clf,
        "bert_clf": bert_clf,
        "transformer_enabled": transformer_enabled,
        "evaluation": evaluation,
    }
    joblib.dump(artifact, MODEL_PATH)

    metrics_data = load_metrics()
    metrics_data["training"] = evaluation
    save_metrics(metrics_data)

    return artifact


def load_metrics() -> Dict:
    if not METRICS_PATH.exists():
        return {"articles_analyzed": 0, "training": {}}
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_metrics(metrics: Dict) -> None:
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


_cached_embedder = None


def get_embedder():
    global _cached_embedder
    if _cached_embedder is None:
        _cached_embedder = _load_sentence_transformer()
    return _cached_embedder


def load_model() -> Dict:
    if not os.path.exists(MODEL_PATH):
        return train_model(DATASET_PATH)
    return joblib.load(MODEL_PATH)


def _combine_probs(artifact: Dict, combined_texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    word_vec = artifact["word_vectorizer"]
    word_clf = artifact["word_clf"]
    char_vec = artifact["char_vectorizer"]
    char_clf = artifact["char_clf"]

    Xw = word_vec.transform(combined_texts)
    Xc = char_vec.transform(combined_texts)

    word_probs_all = word_clf.predict_proba(Xw)
    char_probs_all = char_clf.predict_proba(Xc)

    word_fake = word_probs_all[:, list(word_clf.classes_).index("fake")]
    char_fake = char_probs_all[:, list(char_clf.classes_).index("fake")]

    if artifact.get("transformer_enabled") and artifact.get("bert_clf") is not None:
        embedder = get_embedder()
        if embedder is not None:
            Xb = embedder.encode(combined_texts, batch_size=32, show_progress_bar=False)
            bert_clf = artifact["bert_clf"]
            bert_probs_all = bert_clf.predict_proba(Xb)
            bert_fake = bert_probs_all[:, list(bert_clf.classes_).index("fake")]
            fake_probs = (0.40 * word_fake) + (0.25 * char_fake) + (0.35 * bert_fake)
        else:
            fake_probs = (0.62 * word_fake) + (0.38 * char_fake)
    else:
        fake_probs = (0.62 * word_fake) + (0.38 * char_fake)

    real_probs = 1.0 - fake_probs
    return real_probs, fake_probs


def predict_news(title: str = "", text: str = "", source: str = "") -> Dict:
    artifact = load_model()
    combined_text = build_combined_text(title, text, source)
    if not combined_text:
        raise ValueError("Empty input. Enter title or article text.")

    real_probs, fake_probs = _combine_probs(artifact, [combined_text])
    real_prob = float(real_probs[0])
    fake_prob = float(fake_probs[0])

    prediction = "fake" if fake_prob >= 0.5 else "real"
    confidence = fake_prob if prediction == "fake" else real_prob

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": {
            "real": round(real_prob, 4),
            "fake": round(fake_prob, 4),
        },
    }


def get_top_keywords(title: str = "", text: str = "", source: str = "", top_n: int = 8) -> List[Tuple[str, float]]:
    artifact = load_model()
    vectorizer = artifact["word_vectorizer"]
    clf = artifact["word_clf"]

    combined_text = build_combined_text(title, text, source)
    X = vectorizer.transform([combined_text])
    feature_names = vectorizer.get_feature_names_out()
    row = X.toarray()[0]
    coefficients = clf.coef_[0]

    present_indices = row.nonzero()[0]
    contributions: List[Tuple[str, float]] = []
    for idx in present_indices:
        score = float(row[idx] * coefficients[idx])
        contributions.append((feature_names[idx], score))

    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return contributions[:top_n]


def get_training_metrics() -> Dict:
    metrics = load_metrics().get("training", {})
    if metrics:
        return metrics
    artifact = load_model()
    return artifact.get("evaluation", {})
