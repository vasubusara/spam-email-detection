import re
import joblib

MODEL_PATH = "spam_model.joblib"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model not found. Run: python train.py")
        return

    print("Spam Email Detector (type 'exit' to quit)")
    while True:
        message = input("\nPaste an email/message: ").strip()
        if message.lower() == "exit":
            break
        if not message:
            print("Please enter some text.")
            continue

        prediction = model.predict([clean_text(message)])[0]
        probabilities = model.predict_proba([clean_text(message)])[0]
        confidence = max(probabilities) * 100
        print(f"Prediction: {prediction.upper()} (model confidence: {confidence:.1f}%)")
        print("Note: This is a machine-learning estimate, not a guarantee.")

if __name__ == "__main__":
    main()
