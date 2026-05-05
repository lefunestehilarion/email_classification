#!/usr/bin/env python3
"""
eml_ingest.py — Convertit des .eml en JSON et les classe automatiquement

Usage :
    # Convertir + classer un dossier de .eml (localement)
    python eml_ingest.py --input data/raw_eml/ --output data/inbox/

    # Convertir + envoyer vers l'API à la réception
    python eml_ingest.py --input data/raw_eml/ --api http://localhost:8000

    # Surveiller un dossier en temps réel (mode watch)
    python eml_ingest.py --input data/raw_eml/ --api http://localhost:8000 --watch

    # Lire un .eml depuis stdin (intégration Postfix/procmail)
    cat mail.eml | python eml_ingest.py --stdin --api http://localhost:8000
"""

import argparse
import email
import email.policy
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Conversion EML → dict Python (sans dépendance externe)
# ---------------------------------------------------------------------------
def eml_to_dict(raw: bytes) -> dict:
    """
    Parse un fichier .eml et retourne un dict normalisé
    compatible avec batch_classify.py.
    """
    msg = email.message_from_bytes(raw, policy=email.policy.default)

    # --- Header ---
    subject = str(msg.get("subject", ""))
    from_   = str(msg.get("from", ""))
    to_     = str(msg.get("to", ""))
    date_   = str(msg.get("date", ""))

    # --- Body : extraire text/plain en priorité, sinon text/html ---
    body_plain = ""
    body_html  = ""

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if "attachment" in cd:
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                text    = payload.decode(charset, errors="replace")
                if ct == "text/plain" and not body_plain:
                    body_plain = text
                elif ct == "text/html" and not body_html:
                    body_html = text
            except Exception:
                continue
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset    = msg.get_content_charset() or "utf-8"
            body_plain = payload.decode(charset, errors="replace")

    # Nettoyer le HTML basiquement si pas de plain text
    body = body_plain or _strip_html(body_html)

    return {
        "header": {
            "subject": subject,
            "from":    from_,
            "to":      to_,
            "date":    date_,
        },
        "body": [{"content": body, "content_header": {"content-type": ["text/plain"]}}],
        "_meta": {
            "parsed_at": datetime.utcnow().isoformat(),
            "has_html":  bool(body_html),
            "has_plain": bool(body_plain),
        }
    }


def _strip_html(html: str) -> str:
    """Supprime les balises HTML basiquement."""
    import re
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Envoi vers l'API
# ---------------------------------------------------------------------------
def classify_via_api(subject: str, body: str, api_url: str) -> dict:
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
# Traitement d'un seul .eml
# ---------------------------------------------------------------------------
def process_eml(eml_path: Path, output_dir: Path = None,
                api_url: str = None, model_dir: str = "models",
                _local_predict=None) -> dict:
    """
    Convertit un .eml, le classe, sauvegarde le JSON enrichi.
    Retourne le résultat de classification.
    """
    with open(eml_path, "rb") as f:
        raw = f.read()

    data = eml_to_dict(raw)
    subject = data["header"]["subject"]
    body    = data["body"][0]["content"] if data["body"] else ""

    # Classifier
    if api_url:
        result = classify_via_api(subject, body, api_url)
    elif _local_predict:
        result = _local_predict(subject, body)
    else:
        raise ValueError("Fournir --api ou --model-dir")

    # Enrichir le JSON avec la classification
    data["classification"] = result

    # Sauvegarder le JSON si output_dir fourni
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Organiser par catégorie prédite
        cat_dir = output_dir / result["prediction"]
        cat_dir.mkdir(exist_ok=True)
        out_path = cat_dir / (eml_path.stem + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return result


# ---------------------------------------------------------------------------
# Mode stdin (intégration Postfix / procmail)
# ---------------------------------------------------------------------------
def process_stdin(api_url: str = None, model_dir: str = "models",
                  output_dir: Path = None, _local_predict=None):
    raw = sys.stdin.buffer.read()
    if not raw:
        print("❌ Aucun contenu reçu sur stdin", file=sys.stderr)
        sys.exit(1)

    data    = eml_to_dict(raw)
    subject = data["header"]["subject"]
    body    = data["body"][0]["content"] if data["body"] else ""

    if api_url:
        result = classify_via_api(subject, body, api_url)
    elif _local_predict:
        result = _local_predict(subject, body)
    else:
        raise ValueError("Fournir --api ou --model-dir")

    data["classification"] = result

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        cat_dir = output_dir / result["prediction"]
        cat_dir.mkdir(exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        out_path = cat_dir / f"stdin_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # Sortie JSON sur stdout (utile pour les pipelines)
    print(json.dumps(result, ensure_ascii=False))
    return result


# ---------------------------------------------------------------------------
# Mode batch : dossier de .eml
# ---------------------------------------------------------------------------
def process_folder(input_dir: Path, output_dir: Path = None,
                   api_url: str = None, model_dir: str = "models"):

    eml_files = sorted(input_dir.glob("*.eml"))
    if not eml_files:
        print(f"⚠️  Aucun .eml trouvé dans {input_dir}")
        return

    print(f"📂 {len(eml_files)} fichier(s) .eml trouvé(s)")

    _local_predict = None
    if not api_url:
        _local_predict = _load_local_predictor(model_dir)

    errors = 0
    results = []

    for i, fpath in enumerate(eml_files, 1):
        try:
            result = process_eml(fpath, output_dir, api_url, model_dir, _local_predict)
            results.append((fpath.name, result))
            bar = "█" * int(result["confidence"] * 20)
            print(f"  [{i:>4}/{len(eml_files)}] {fpath.name:<35} "
                  f"→ {result['prediction']:<25} {bar} {result['confidence']:.2%}")
        except Exception as e:
            print(f"  ❌ {fpath.name} — {e}")
            errors += 1

    print(f"\n✅ Traités : {len(results)} | ❌ Erreurs : {errors}")
    if output_dir:
        print(f"💾 JSON classifiés dans : {output_dir}/{{categorie}}/")


# ---------------------------------------------------------------------------
# Mode watch : surveillance en temps réel
# ---------------------------------------------------------------------------
def watch_folder(input_dir: Path, output_dir: Path = None,
                 api_url: str = None, model_dir: str = "models",
                 interval: int = 5):

    print(f"👁️  Surveillance de {input_dir} (intervalle : {interval}s) — Ctrl+C pour arrêter")

    _local_predict = None
    if not api_url:
        _local_predict = _load_local_predictor(model_dir)

    seen = set()

    try:
        while True:
            eml_files = set(input_dir.glob("*.eml"))
            new_files = eml_files - seen

            for fpath in sorted(new_files):
                try:
                    result = process_eml(fpath, output_dir, api_url, model_dir, _local_predict)
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{ts}] {fpath.name:<35} → {result['prediction']} "
                          f"({result['confidence']:.2%})")
                    seen.add(fpath)
                except Exception as e:
                    print(f"  ❌ {fpath.name} — {e}")
                    seen.add(fpath)  # éviter de réessayer en boucle

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n👋 Surveillance arrêtée.")


# ---------------------------------------------------------------------------
# Chargement du prédicteur local
# ---------------------------------------------------------------------------
def _load_local_predictor(model_dir: str):
    try:
        import pickle
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import nltk, string
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer

        for pkg in ['stopwords', 'punkt', 'punkt_tab']:
            nltk.download(pkg, quiet=True)

        ps = PorterStemmer()

        def transform_text(text):
            tokens = nltk.word_tokenize(text.lower())
            tokens = [t for t in tokens if t.isalnum()
                      and t not in stopwords.words('english')
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
            proba    = model.predict(X, verbose=0)[0]
            pred_idx = int(np.argmax(proba))
            top3_idx = np.argsort(proba)[::-1][:3]
            return {
                "prediction": encoder.classes_[pred_idx],
                "confidence": round(float(proba[pred_idx]), 4),
                "top_k": [
                    {"label": encoder.classes_[i], "confidence": round(float(proba[i]), 4)}
                    for i in top3_idx
                ]
            }

        print(f"✅ Modèle local chargé ({meta['num_classes']} catégories)")
        return predict_local

    except Exception as e:
        print(f"❌ Impossible de charger le modèle local : {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convertit des .eml en JSON et les classe automatiquement"
    )
    parser.add_argument("--input",      help="Dossier contenant les .eml")
    parser.add_argument("--output",     default=None,
                        help="Dossier de sortie pour les JSON classifiés")
    parser.add_argument("--api",        default=None,
                        help="URL de l'API REST (ex: http://localhost:8000)")
    parser.add_argument("--model-dir",  default="models")
    parser.add_argument("--watch",      action="store_true",
                        help="Surveiller le dossier en temps réel")
    parser.add_argument("--interval",   type=int, default=5,
                        help="Intervalle de surveillance en secondes (défaut: 5)")
    parser.add_argument("--stdin",      action="store_true",
                        help="Lire un .eml depuis stdin (intégration Postfix)")

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    # Mode stdin
    if args.stdin:
        process_stdin(
            api_url=args.api,
            model_dir=args.model_dir,
            output_dir=output_dir
        )

    # Mode dossier
    elif args.input:
        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"❌ Dossier introuvable : {input_dir}")
            sys.exit(1)

        if args.watch:
            watch_folder(input_dir, output_dir, args.api, args.model_dir, args.interval)
        else:
            process_folder(input_dir, output_dir, args.api, args.model_dir)

    else:
        parser.print_help()
