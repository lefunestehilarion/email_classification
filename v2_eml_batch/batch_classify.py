#!/usr/bin/env python3
"""
batch_classify.py — Classe un dossier de fichiers JSON (issus de .eml convertis)

Usage :
    # Classer tous les JSON d'un dossier, résultat dans un CSV
    python batch_classify.py --input data/inbox/ --output results/classified.csv

    # Classer un seul fichier JSON
    python batch_classify.py --input data/inbox/mail_001.json

    # Via l'API (si elle tourne)
    python batch_classify.py --input data/inbox/ --api http://localhost:8000
"""

import argparse
import json
import os
import glob
import csv
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Extraction subject + body depuis un JSON issu de eml_parser
# Compatible aussi avec les formats "plats" maison
# ---------------------------------------------------------------------------
def extract_fields(data: dict) -> tuple[str, str]:
    """
    Extrait (subject, body) depuis un JSON eml_parser ou format plat.
    Gère les deux structures courantes.
    """
    subject = ""
    body    = ""

    # --- Format eml_parser (structure imbriquée) ---
    if "header" in data:
        subject = data["header"].get("subject", "")
        body_parts = data.get("body", [])
        if isinstance(body_parts, list):
            # Prendre la première partie text/plain, sinon la première dispo
            for part in body_parts:
                ct = ""
                if "content_header" in part:
                    ct_list = part["content_header"].get("content-type", [""])
                    ct = ct_list[0] if ct_list else ""
                content = part.get("content", "")
                if "text/plain" in ct:
                    body = content
                    break
                elif not body and content:
                    body = content

    # --- Format plat (subject/body directs) ---
    elif "subject" in data or "body" in data:
        subject = data.get("subject", "")
        raw_body = data.get("body", "")
        # body peut être une liste de parts ou une string
        if isinstance(raw_body, list):
            for part in raw_body:
                if isinstance(part, dict):
                    body = part.get("content", part.get("text", ""))
                elif isinstance(part, str):
                    body = part
                if body:
                    break
        else:
            body = str(raw_body)

    # --- Format MailToJson (parts) ---
    elif "parts" in data:
        subject = data.get("subject", "")
        for part in data.get("parts", []):
            if "text/plain" in part.get("content_type", ""):
                body = part.get("content", "")
                break
        if not body and data.get("parts"):
            body = data["parts"][0].get("content", "")

    return str(subject), str(body)


# ---------------------------------------------------------------------------
# Prédiction locale (sans API)
# ---------------------------------------------------------------------------
def load_local_predictor(model_dir: str = "models"):
    """Charge les artefacts locaux une seule fois."""
    try:
        import pickle, numpy as np
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import nltk, string
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer

        for pkg in ['stopwords', 'punkt', 'punkt_tab']:
            nltk.download(pkg, quiet=True)

        ps = PorterStemmer()

        def transform_text(text):
            text = text.lower()
            tokens = nltk.word_tokenize(text)
            tokens = [t for t in tokens if t.isalnum()]
            tokens = [t for t in tokens if t not in stopwords.words('english')
                      and t not in string.punctuation]
            return " ".join([ps.stem(t) for t in tokens])

        with open(os.path.join(model_dir, "keras_tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)
        with open(os.path.join(model_dir, "meta.json")) as f:
            meta = json.load(f)

        model   = tf.keras.models.load_model(os.path.join(model_dir, "lstm_model.keras"))
        max_len = meta["max_len"]

        def predict_local(subject: str, body: str) -> dict:
            raw       = (subject + " " + body).strip()
            processed = transform_text(raw)
            seq = tokenizer.texts_to_sequences([processed])
            X   = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
            proba     = model.predict(X, verbose=0)[0]
            pred_idx  = int(np.argmax(proba))
            top3_idx  = np.argsort(proba)[::-1][:3]
            return {
                "prediction": encoder.classes_[pred_idx],
                "confidence": round(float(proba[pred_idx]), 4),
                "top_k": [
                    {"label": encoder.classes_[i], "confidence": round(float(proba[i]), 4)}
                    for i in top3_idx
                ]
            }

        print(f"✅ Modèle local chargé — {meta['num_classes']} catégories")
        return predict_local

    except Exception as e:
        print(f"❌ Impossible de charger le modèle local : {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Prédiction via API
# ---------------------------------------------------------------------------
def predict_via_api(subject: str, body: str, api_url: str) -> dict:
    import urllib.request
    payload = json.dumps({"subject": subject, "body": body, "top_k": 3}).encode()
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Traitement batch
# ---------------------------------------------------------------------------
def process_batch(input_path: str, output_csv: str = None,
                  model_dir: str = "models", api_url: str = None):

    input_path = Path(input_path)

    # Collecter les fichiers JSON
    if input_path.is_dir():
        files = sorted(input_path.glob("*.json"))
    elif input_path.suffix == ".json":
        files = [input_path]
    else:
        print(f"❌ {input_path} n'est ni un dossier ni un fichier .json")
        sys.exit(1)

    if not files:
        print(f"⚠️  Aucun fichier JSON trouvé dans {input_path}")
        sys.exit(0)

    print(f"📂 {len(files)} fichier(s) trouvé(s)")

    # Charger le prédicteur
    if api_url:
        print(f"🌐 Mode API : {api_url}")
        def predict_fn(subject, body):
            return predict_via_api(subject, body, api_url)
    else:
        predict_fn = load_local_predictor(model_dir)

    # Préparer la sortie CSV
    results = []
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        csv_file = open(output_csv, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=[
            "file", "subject", "prediction", "confidence",
            "top1", "top1_conf", "top2", "top2_conf", "top3", "top3_conf"
        ])
        writer.writeheader()
    else:
        csv_file = writer = None

    errors = 0
    for i, fpath in enumerate(files, 1):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            subject, body = extract_fields(data)

            if not subject and not body:
                print(f"  ⚠️  [{i}/{len(files)}] {fpath.name} — champs vides, ignoré")
                errors += 1
                continue

            result = predict_fn(subject, body)

            top_k = result.get("top_k", [])
            row = {
                "file":       fpath.name,
                "subject":    subject[:80],
                "prediction": result["prediction"],
                "confidence": f"{result['confidence']:.2%}",
                "top1":       top_k[0]["label"]      if len(top_k) > 0 else "",
                "top1_conf":  f"{top_k[0]['confidence']:.2%}" if len(top_k) > 0 else "",
                "top2":       top_k[1]["label"]      if len(top_k) > 1 else "",
                "top2_conf":  f"{top_k[1]['confidence']:.2%}" if len(top_k) > 1 else "",
                "top3":       top_k[2]["label"]      if len(top_k) > 2 else "",
                "top3_conf":  f"{top_k[2]['confidence']:.2%}" if len(top_k) > 2 else "",
            }

            results.append(row)
            if writer:
                writer.writerow(row)

            # Affichage console
            bar = "█" * int(result["confidence"] * 20)
            print(f"  [{i:>4}/{len(files)}] {fpath.name:<30} "
                  f"→ {result['prediction']:<25} {bar} {result['confidence']:.2%}")

        except Exception as e:
            print(f"  ❌ [{i}/{len(files)}] {fpath.name} — Erreur : {e}")
            errors += 1

    if csv_file:
        csv_file.close()

    # Résumé
    print(f"\n{'─'*60}")
    print(f"✅ Traités : {len(results)} | ❌ Erreurs : {errors}")

    if results:
        from collections import Counter
        counts = Counter(r["prediction"] for r in results)
        print(f"\n📊 Distribution des catégories :")
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * count
            print(f"   {label:<25} {bar} ({count})")

    if output_csv:
        print(f"\n💾 Résultats sauvegardés dans : {output_csv}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifieur batch d'emails JSON")
    parser.add_argument("--input",      required=True,
                        help="Dossier ou fichier JSON à traiter")
    parser.add_argument("--output",     default=None,
                        help="Fichier CSV de sortie (ex: results/classified.csv)")
    parser.add_argument("--model-dir",  default="models",
                        help="Dossier contenant les artefacts du modèle")
    parser.add_argument("--api",        default=None,
                        help="URL de l'API REST (ex: http://localhost:8000)")
    args = parser.parse_args()

    process_batch(
        input_path=args.input,
        output_csv=args.output,
        model_dir=args.model_dir,
        api_url=args.api
    )
