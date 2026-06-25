import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("dataset/emails.csv")

# Feature extraction
def preprocess(text):
    text = str(text).lower()

    # URL detection
    text = re.sub(r"http\S+|www\S+", " URL ", text)

    return text

data["text"] = data["text"].apply(preprocess)

X = data["text"]
y = data["label"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Testing
while True:
    email = input("\nEnter Email Text (or type exit): ")

    if email.lower() == "exit":
        break

    email = preprocess(email)

    features = vectorizer.transform([email])

    prediction = model.predict(features)

    if prediction[0] == 1:
        print("⚠️ PHISHING EMAIL")
    else:
        print("✅ SAFE EMAIL")