# Spam Email Detection System

A beginner-friendly machine-learning project that classifies email/message text as **spam** or **ham (not spam)** using TF-IDF features and Logistic Regression.

## Features
- Text cleaning and normalization
- TF-IDF word and phrase features
- Logistic Regression classifier
- Train/test split with stratification
- Accuracy, precision, recall, F1-score and confusion matrix
- Interactive command-line prediction

## Project structure

```text
spam-email-detection/
├── data/
│   └── emails.csv
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

The script trains the model, prints evaluation metrics, and saves `spam_model.joblib`.

## Test a message

```bash
python predict.py
```

Paste an email or message when prompted. Type `exit` to stop.

## Dataset

`data/emails.csv` is a small illustrative starter dataset included so the project runs immediately. It is **not** a large or representative email corpus, and the resulting scores should not be treated as evidence of real-world performance.

For a meaningful evaluation, replace it with a properly licensed, larger labelled dataset. Keep the same CSV format:

```csv
text,label
"Your message text here",ham
"Your message text here",spam
```

Labels must be `spam` or `ham`. Avoid putting private or real personal emails into a public repository.

## How it works

1. **Cleaning:** normalizes text, URLs and email addresses.
2. **TF-IDF:** converts text into numerical features based on word importance.
3. **Logistic Regression:** learns to classify the features as spam or ham.
4. **Evaluation:** tests the model on a held-out portion of the dataset.

## Limitations

- A small starter dataset cannot capture the variety of real spam.
- The model may make false-positive and false-negative predictions.
- The confidence score is not a calibrated probability.
- Do not use this as the sole filter for important email.

## GitHub upload

Create a new public repository named `spam-email-detection`, then upload these project files. Do not upload `.venv/` or private datasets.

## Author

Add your name and a short description of your actual contribution here. If this project is adapted from a team or other source, acknowledge that contribution and follow its licence.
