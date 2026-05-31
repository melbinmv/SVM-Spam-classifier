# 📧 Spam Detector

A web app that classifies SMS or email messages as **spam** or **ham** (not spam) using a trained Support Vector Machine (SVM) model, served via a FastAPI backend.

---

## Features

- Paste any message and instantly get a spam/ham prediction
- Displays the **probability** of the message being spam
- Clean web interface built with Jinja2 HTML templates
- Lightweight ML pipeline using a pre-trained SVM + CountVectorizer

---

## Project Structure

```
├── main.py                 # FastAPI application
├── svm_model.pkl           # Trained SVM model
├── cv.joblib               # Fitted CountVectorizer
├── templates/
│   └── index.html          # Frontend HTML template
└── README.md
```

---

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- scikit-learn
- joblib
- nltk
- python-multipart

Install dependencies:

```bash
pip install fastapi uvicorn scikit-learn joblib nltk python-multipart
```

---

## Running the App

```bash
uvicorn main:app --reload
```

Then open your browser at `http://127.0.0.1:8000`.

---

## How It Works

1. The user submits a message via the web form.
2. The message is vectorized using the pre-fitted `CountVectorizer` (`cv.joblib`).
3. The SVM model (`svm_model.pkl`) predicts whether the message is spam or ham.
4. The app also returns the **probability of being spam** using `predict_proba`.
5. The result is rendered back on the same page.

---

## Model Details

| Component     | Details                        |
|---------------|--------------------------------|
| Algorithm     | Support Vector Machine (SVM)   |
| Vectorizer    | CountVectorizer                |
| Output labels | `ham` (not spam), `spam`       |

> The model was trained on a labelled SMS/email dataset. A commented-out preprocessing pipeline (stopword removal + stemming via NLTK's PorterStemmer) is included in the code if you wish to extend the pipeline.

---

## Example Output

| Input Message | Prediction | Spam Probability |
|---|---|---|
| "Hey, are we still on for lunch?" | Not likely a Spam! | 0.03 |
| "WINNER! Claim your free prize now" | Likely a Spam! | 0.97 |
