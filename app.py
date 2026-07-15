from flask import Flask, render_template, request, redirect
import joblib
import sqlite3

app = Flask(__name__)

# Load AI Model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def get_db():
    return sqlite3.connect("history.db")


@app.route("/")
def home():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    history = cursor.fetchall()

    total_predictions = len(history)

    conn.close()

    return render_template(
        "index.html",
        history=history,
        total_predictions=total_predictions
    )


@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)[0]

    confidence = round(model.predict_proba(vector).max() * 100, 2)

    if prediction == 1:
        result = "✅ Real News"
    else:
        result = "❌ Fake News"

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO history(news, prediction, confidence) VALUES (?, ?, ?)",
        (news, result, confidence)
    )

    conn.commit()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    history = cursor.fetchall()

    total_predictions = len(history)

    conn.close()

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        news=news,
        history=history,
        total_predictions=total_predictions
    )


@app.route("/clear")
def clear():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM history")

    conn.commit()
    conn.close()

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)