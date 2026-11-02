import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/emails.csv"
MODEL_PATH = "spam_model.joblib"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    data = pd.read_csv(DATA_PATH)
    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    data = data.dropna(subset=["text", "label"]).copy()
    data["label"] = data["label"].str.lower().str.strip()
    data = data[data["label"].isin(["spam", "ham"])]
    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain both 'spam' and 'ham' labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"].map(clean_text), data["label"],
        test_size=0.25, random_state=42, stratify=data["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                                 stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix [ham, spam]:")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
