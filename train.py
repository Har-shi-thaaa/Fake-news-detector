import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("Loading dataset...")

# Read datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Label the data
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Keep only title and text
data = data[["title", "text", "label"]]

# Combine title + text
data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")

# Features and labels
X = data["content"]
y = data["label"]

print("Converting text into vectors...")

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_vector = vectorizer.fit_transform(X)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42
)

print("Training AI model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print(f"Model Accuracy: {accuracy*100:.2f}%")

# Save model
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model Saved Successfully!")