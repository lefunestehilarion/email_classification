#!/usr/bin/env python3
"""
predict.py — Prédiction CLI et API REST (FastAPI)

--- CLI ---
    python predict.py --subject "Budget Q3" --body "Please find attached..."
    python predict.py --text "Reunion equipe demain 9h salle B"

--- API ---
    uvicorn predict:app --host 0.0.0.0 --port 8000
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"subject": "Budget Q3", "body": "See attached report."}'
"""

import argparse
import os
import pickle
import json
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

for pkg in ['stopwords', 'punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

ps = PorterStemmer()

# ---------------------------------------------------------------------------
# Préprocessing (identique à train.py)
# ---------------------------------------------------------------------------
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Chargement des artefacts (singleton)
# ---------------------------------------------------------------------------
_artifacts = {}

def load_artifacts(model_dir: str = "models"):
    global _artifacts
    if _artifacts:
        return _artifacts

    print(f"📦 Chargement des artefacts depuis '{model_dir}'...")

    with open(os.path.join(model_dir, "keras_tokenizer.pkl"), "rb") as f:
        _artifacts["tokenizer"] = pickle.load(f)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        _artifacts["encoder"] = pickle.load(f)
    with open(os.path.join(model_dir, "meta.json")) as f:
        _artifacts["meta"] = json.load(f)

    _artifacts["model"] = tf.keras.models.load_model(
        os.path.join(model_dir, "lstm_model.keras")
    )
    print(f"✅ Modèle chargé — {_artifacts['meta']['num_classes']} catégories : "
          f"{_artifacts['meta']['categories']}")
    return _artifacts


# ---------------------------------------------------------------------------
# Prédiction core
# ---------------------------------------------------------------------------
def predict(subject: str = "", body: str = "", text: str = "",
            model_dir: str = "models", top_k: int = 3) -> dict:
    """
    Retourne la catégorie prédite + le top-k des probabilités.
    """
    arts = load_artifacts(model_dir)
    tokenizer = arts["tokenizer"]
    encoder   = arts["encoder"]
    model     = arts["model"]
    max_len   = arts["meta"]["max_len"]

    # Construire le texte brut
    raw = text if text else (subject + " " + body)
    processed = transform_text(raw)

    # Tokeniser et padder
    seq = tokenizer.texts_to_sequences([processed])
    X   = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Prédire
    proba = model.predict(X, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = encoder.classes_[pred_idx]

    # Top-k
    top_indices = np.argsort(proba)[::-1][:top_k]
    top_k_results = [
        {"label": encoder.classes_[i], "confidence": round(float(proba[i]), 4)}
        for i in top_indices
    ]

    return {
        "prediction": pred_label,
        "confidence": round(float(proba[pred_idx]), 4),
        "top_k": top_k_results
    }


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(
        title="Email Classifier API",
        description="Classifie un email parmi N catégories via un modèle LSTM.",
        version="1.0.0"
    )

    MODEL_DIR = os.environ.get("MODEL_DIR", "models")

    class EmailRequest(BaseModel):
        subject: str = ""
        body:    str = ""
        text:    str = ""
        top_k:   int = 3

    class PredictionResponse(BaseModel):
        prediction: str
        confidence: float
        top_k:      list

    @app.on_event("startup")
    def startup_event():
        load_artifacts(MODEL_DIR)

    @app.get("/health")
    def health():
        arts = load_artifacts(MODEL_DIR)
        return {
            "status": "ok",
            "categories": arts["meta"]["categories"],
            "num_classes": arts["meta"]["num_classes"]
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict_endpoint(req: EmailRequest):
        return predict(
            subject=req.subject,
            body=req.body,
            text=req.text,
            model_dir=MODEL_DIR,
            top_k=req.top_k
        )

except ImportError:
    # FastAPI non installé : mode CLI uniquement
    app = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifieur d'emails — prédiction")
    parser.add_argument("--subject",    default="", help="Objet de l'email")
    parser.add_argument("--body",       default="", help="Corps de l'email")
    parser.add_argument("--text",       default="", help="Texte brut (alternative à subject+body)")
    parser.add_argument("--model-dir",  default="models")
    parser.add_argument("--top-k",      type=int, default=3)
    args = parser.parse_args()

    if not args.text and not args.subject and not args.body:
        parser.error("Fournir --text ou --subject/--body")

    result = predict(
        subject=args.subject,
        body=args.body,
        text=args.text,
        model_dir=args.model_dir,
        top_k=args.top_k
    )

    print(f"\n📧 Prédiction  : {result['prediction']}")
    print(f"🎯 Confiance   : {result['confidence']:.2%}")
    print(f"\n📊 Top {args.top_k} :")
    for r in result["top_k"]:
        bar = "█" * int(r["confidence"] * 30)
        print(f"   {r['label']:<25} {bar} {r['confidence']:.2%}")
