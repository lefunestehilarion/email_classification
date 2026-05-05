#!/usr/bin/env python3
"""
classify_csv.py — Classe un CSV d'emails (subject + body) sans label

Usage :
    # Localement (modèle chargé depuis models/)
    python classify_csv.py --input data/emails.csv

    # Via l'API
    python classify_csv.py --input data/emails.csv --api http://localhost:8000

    # Spécifier le fichier de sortie
    python classify_csv.py --input data/emails.csv --output results/classified.csv

Format CSV attendu :
    subject,body
    "Budget Q3","Please find attached..."
    "Match results","Liverpool won..."
"""

import argparse
import os
import sys
import json
import csv
import pickle
import numpy as np
import pandas as pd
import urllib.request
import nltk
import string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

for pkg in ['stopwords', 'punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

ps = PorterStemmer()


# ---------------------------------------------------------------------------
# Préprocessing
# ---------------------------------------------------------------------------
def transform_text(text: str) -> str:
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum()
              and t not in stopwords.words('english')
              and t not in string.punctuation]
    return " ".join([ps.stem(t) for t in tokens])


# ---------------------------------------------------------------------------
# Chargement du modèle local
# ---------------------------------------------------------------------------
def load_local_predictor(model_dir: str = "models"):
    with open(os.path.join(model_dir, "keras_tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)

    model   = tf.keras.models.load_model(os.path.join(model_dir, "lstm_model.keras"))
    max_len = meta["max_len"]

    print(f"✅ Modèle chargé — {meta['num_classes']} catégories : {meta['categories']}")

    def predict_batch(subjects: list, bodies: list) -> list:
        texts     = [transform_text(s + " " + b) for s, b in zip(subjects, bodies)]
        seqs      = tokenizer.texts_to_sequences(texts)
        X         = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
        probas    = model.predict(X, verbose=0)
        results   = []
        for proba in probas:
            pred_idx = int(np.argmax(proba))
            top3_idx = np.argsort(proba)[::-1][:3]
            results.append({
                "prediction": encoder.classes_[pred_idx],
                "confidence": round(float(proba[pred_idx]), 4),
                "top2": encoder.classes_[top3_idx[1]],
                "top2_conf": round(float(proba[top3_idx[1]]), 4),
                "top3": encoder.classes_[top3_idx[2]],
                "top3_conf": round(float(proba[top3_idx[2]]), 4),
            })
        return results

    return predict_batch


# ---------------------------------------------------------------------------
# Prédiction via API (une par une)
# ---------------------------------------------------------------------------
def predict_via_api(subject: str, body: str, api_url: str) -> dict:
    payload = json.dumps({"subject": subject, "body": body, "top_k": 3}).encode()
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        r = json.loads(resp.read())
        top_k = r.get("top_k", [])
        return {
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "top2":       top_k[1]["label"]      if len(top_k) > 1 else "",
            "top2_conf":  top_k[1]["confidence"] if len(top_k) > 1 else 0,
            "top3":       top_k[2]["label"]      if len(top_k) > 2 else "",
            "top3_conf":  top_k[2]["confidence"] if len(top_k) > 2 else 0,
        }


# ---------------------------------------------------------------------------
# Classification du CSV
# ---------------------------------------------------------------------------
def classify_csv(input_path: str, output_path: str = None,
                 model_dir: str = "models", api_url: str = None,
                 batch_size: int = 64):

    # --- Charger le CSV ---
    print(f"📂 Chargement de {input_path}...")
    df = pd.read_csv(input_path, encoding='utf-8')

    # Normaliser les noms de colonnes (minuscules, strip)
    df.columns = df.columns.str.strip().str.lower()

    if "subject" not in df.columns and "body" not in df.columns:
        print(f"❌ Colonnes trouvées : {list(df.columns)}")
        print("   Le CSV doit contenir 'subject' et/ou 'body'")
        sys.exit(1)

    df["subject"] = df.get("subject", pd.Series([""] * len(df))).fillna("")
    df["body"]    = df.get("body",    pd.Series([""] * len(df))).fillna("")

    print(f"✅ {len(df)} emails à classifier")

    # --- Classifier ---
    predictions = []

    if api_url:
        print(f"🌐 Mode API : {api_url}")
        for i, row in df.iterrows():
            try:
                result = predict_via_api(row["subject"], row["body"], api_url)
                predictions.append(result)
                if (i + 1) % 50 == 0:
                    print(f"   {i+1}/{len(df)} traités...")
            except Exception as e:
                print(f"  ⚠️  Ligne {i} — erreur : {e}")
                predictions.append({
                    "prediction": "error", "confidence": 0,
                    "top2": "", "top2_conf": 0,
                    "top3": "", "top3_conf": 0
                })
    else:
        predict_batch = load_local_predictor(model_dir)
        print(f"🚀 Classification par batch de {batch_size}...")
        for start in range(0, len(df), batch_size):
            end     = min(start + batch_size, len(df))
            batch   = df.iloc[start:end]
            results = predict_batch(
                batch["subject"].tolist(),
                batch["body"].tolist()
            )
            predictions.extend(results)
            print(f"   {end}/{len(df)} traités...")

    # --- Construire le dataframe de sortie ---
    results_df = pd.DataFrame(predictions)
    output_df  = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    # Formater la confiance en pourcentage lisible
    output_df["confidence"] = output_df["confidence"].apply(lambda x: f"{x:.2%}")
    output_df["top2_conf"]  = output_df["top2_conf"].apply(lambda x: f"{x:.2%}")
    output_df["top3_conf"]  = output_df["top3_conf"].apply(lambda x: f"{x:.2%}")

    # --- Sauvegarder ---
    if not output_path:
        input_stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{input_stem}_classified.csv")

    output_df.to_csv(output_path, index=False, encoding='utf-8')

    # --- Résumé console ---
    print(f"\n{'─'*60}")
    print(f"✅ {len(output_df)} emails classifiés")
    print(f"💾 Résultat sauvegardé : {output_path}")

    from collections import Counter
    counts = Counter(r["prediction"] for r in predictions if r["prediction"] != "error")
    print(f"\n📊 Distribution des catégories :")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {label:<25} {bar} {count} ({pct:.1f}%)")

    return output_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifie un CSV d'emails sans label")
    parser.add_argument("--input",      required=True, help="CSV à classifier (subject, body)")
    parser.add_argument("--output",     default=None,  help="CSV de sortie (défaut: input_classified.csv)")
    parser.add_argument("--model-dir",  default="models")
    parser.add_argument("--api",        default=None,  help="URL de l'API REST")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    classify_csv(
        input_path=args.input,
        output_path=args.output,
        model_dir=args.model_dir,
        api_url=args.api,
        batch_size=args.batch_size
    )
