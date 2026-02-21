from flask import Flask, render_template, request
import sqlite3
import spacy

app = Flask(__name__)
nlp = spacy.load("en_core_web_md")

def process_text(text):
    doc = nlp(text)

    original_tokens = [token.text.lower() for token in doc]

    cleaned_tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    return original_tokens, cleaned_tokens, doc


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    username = request.form["username"]
    text = request.form["text"]

    conn = sqlite3.connect("nlp.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO users (username) VALUES (?)", (username,))
    user_id = cur.lastrowid

    original_tokens, cleaned_tokens, doc = process_text(text)

   
    cur.execute(
        "INSERT INTO processed_text (user_id, original_tokens, cleaned_tokens) VALUES (?, ?, ?)",
        (user_id, str(original_tokens), str(cleaned_tokens))
    )
    text_id = cur.lastrowid
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    for ent_text, ent_label in entities:
        cur.execute(
            "INSERT INTO entities (text_id, entity_text, entity_label) VALUES (?, ?, ?)",
            (text_id, ent_text, ent_label)
        )

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        original_tokens=original_tokens,
        cleaned_tokens=cleaned_tokens,
        entities=entities
    )


@app.route("/database")
def database():

    conn = sqlite3.connect("nlp.db")
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            users.username,
            processed_text.original_tokens,
            processed_text.cleaned_tokens,
            GROUP_CONCAT(entities.entity_text || ' (' || entities.entity_label || ')')
        FROM processed_text
        JOIN users 
            ON users.user_id = processed_text.user_id
        LEFT JOIN entities 
            ON entities.text_id = processed_text.text_id
        GROUP BY processed_text.text_id
        ORDER BY processed_text.text_id DESC
    """)

    records = cur.fetchall()

    conn.close()

    return render_template("db.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)
    print("server started")