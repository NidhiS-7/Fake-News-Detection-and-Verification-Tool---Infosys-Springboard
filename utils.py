import re
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_combined_text(title: str = "", text: str = "", source: str = "") -> str:
    title = clean_text(title)
    text = clean_text(text)
    source = clean_text(source)

    parts = []
    if title:
        parts.append(f"title {title}")
    if source:
        parts.append(f"source {source}")
    if text:
        parts.append(f"body {text}")

    return " ".join(parts).strip()
