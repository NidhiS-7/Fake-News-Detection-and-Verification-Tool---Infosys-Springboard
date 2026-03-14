SUSPICIOUS_KEYWORDS = [
    "miracle",
    "cure",
    "instantly",
    "regrow",
    "secret",
    "magic",
    "guaranteed",
    "ancient",
    "hidden",
    "shocking",
    "viral",
    "leaked",
    "exposed",
    "doctors hate",
    "one leaf",
    "one herb",
    "all diseases",
    "cancer instantly",
]


def find_suspicious_claims(text):
    text = text.lower()
    found = []

    for word in SUSPICIOUS_KEYWORDS:
        if word in text:
            found.append(word)

    return found


def generate_explanation(title, text, source, prediction, confidence, important_terms):
    combined_text = f"{title} {text}".strip()
    suspicious_claims = find_suspicious_claims(combined_text)

    explanation = []

    if suspicious_claims:
        explanation.append(
            f"Suspicious words/phrases detected: {', '.join(suspicious_claims)}."
        )

    if important_terms:
        explanation.append(
            f"Top terms influencing the decision: {', '.join(important_terms)}."
        )

    if source and source.strip():
        explanation.append(
            f"The entered source was '{source}', and source credibility is checked separately in the dashboard."
        )

    if prediction == "fake":
        explanation.append(
            f"The combined signal leans FAKE with confidence {confidence:.2f} because the language looks exaggerated, sensational, or misleading."
        )
    else:
        explanation.append(
            f"The combined signal leans REAL with confidence {confidence:.2f} because the writing looks more neutral, factual, and report-like."
        )

    return explanation