from flask import Flask, render_template, request
from transformers import pipeline
import requests

app = Flask(__name__)
classifier = pipeline("text-classification", model="hamzab/roberta-fake-news-classification")

def classify_news(text):
    result = classifier(text, truncation=True, max_length=512)[0]
    print(f"--- DIAGNOSTIC: {result} ---")

    raw_label = result['label'].upper()
    score = result['score']
    #LABEL_1 = REAL, LABEL_0 = FAKE
    if raw_label == "LABEL_1" or "TRUE" in raw_label:
        final_label = "REAL"
    else:
        final_label = "FAKE"

    return final_label, round(score * 100, 2)

def verify_claim(text):
    claim = " ".join(text.split()[:6])
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": claim, "format": "json"}
    headers = {'User-Agent': 'NewsApp/1.0'}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        if search_results:
            return f"Found on Wiki: {search_results[0]['title']}"
        return "No matching factual records."
    except:
        return "Verification service offline."

@app.route("/", methods=["GET", "POST"])
def index():
    data = {"result": None, "confidence": 0, "verification": None, "news_text": ""}
    if request.method == "POST":
        news_text = request.form.get("news")
        if news_text:
            label, conf = classify_news(news_text)
            verif = verify_claim(news_text)
            data.update({"result": label, "confidence": conf, "verification": verif, "news_text": news_text})
    return render_template("index.html", **data)

if __name__ == "__main__":
    app.run(debug=True)