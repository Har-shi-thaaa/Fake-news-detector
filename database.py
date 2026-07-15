import sqlite3

conn = sqlite3.connect("history.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
id INTEGER PRIMARY KEY AUTOINCREMENT,
news TEXT,
prediction TEXT,
confidence REAL
)
""")

conn.commit()

conn.close()

print("Database Created Successfully!")