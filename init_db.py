import sqlite3

conn = sqlite3.connect("nlp.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users(
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS processed_text(
    text_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    original_tokens TEXT,
    cleaned_tokens TEXT,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS entities(
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_id INTEGER,
    entity_text TEXT,
    entity_label TEXT,
    FOREIGN KEY(text_id) REFERENCES processed_text(text_id)
)
""")

conn.commit()
conn.close()

print("Database created successfully ✅")
