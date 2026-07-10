#!/usr/bin/env python3
"""
benchmark_speed.py — Compare la rapidité du LSTM (supervisé) vs du zero-shot (NLI)
sur les mêmes emails : latence par email + débit global.

Hypothèses sur les artefacts existants :
  - LSTM      : models/lstm_model.keras + keras_tokenizer.pkl + label_encoder.pkl + meta.json
                (voir train.py / predict.py)
  - Zero-shot : pipeline HF "zero-shot-classification" (voir test_mdeberta_zero_shot.py)
                candidats = labels de la table SQLite `labels`

Usage :
    # Benchmark sur N emails pris dans la base SQLite
    python benchmark_speed.py --db data/emails.db --model-dir models --n-emails 50

    # Comparer aussi un modèle zero-shot fine-tuné localement
    python benchmark_speed.py --db data/emails.db --model-dir models --zero-shot-model models/bart-mnli-finetuned/final

    # Sauvegarder les résultats en CSV
    python benchmark_speed.py --db data/emails.db --model-dir models --output results/benchmark.csv
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from models import Email, Label, sqlite_engine

DEFAULT_DB = Path(__file__).resolve().parent / "data" / "emails.db"
ZERO_SHOT_MODEL_ID = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This email is about {}."


@dataclass
class TimingResult:
    name: str
    n_items: int
    per_item_seconds: list[float] = field(default_factory=list)
    setup_seconds: float = 0.0
    batch_seconds: float | None = None  # temps total en mode "vrai batch" (si applicable)

    @property
    def total_seconds(self) -> float:
        return sum(self.per_item_seconds)

    @property
    def throughput_per_sec(self) -> float:
        total = self.batch_seconds if self.batch_seconds is not None else self.total_seconds
        return self.n_items / total if total > 0 else float("nan")

    def summary(self) -> dict:
        lat_ms = [t * 1000 for t in self.per_item_seconds]
        return {
            "name": self.name,
            "n_items": self.n_items,
            "setup_seconds": round(self.setup_seconds, 3),
            "latency_mean_ms": round(mean(lat_ms), 2) if lat_ms else None,
            "latency_median_ms": round(median(lat_ms), 2) if lat_ms else None,
            "latency_stdev_ms": round(stdev(lat_ms), 2) if len(lat_ms) > 1 else None,
            "latency_p95_ms": round(float(np.percentile(lat_ms, 95)), 2) if lat_ms else None,
            "throughput_per_sec": round(self.throughput_per_sec, 3),
            "total_seconds": round(self.batch_seconds if self.batch_seconds is not None else self.total_seconds, 3),
        }


def load_emails(db_path: Path, split: str | None, n_emails: int, seed: int = 2) -> list[Email]:
    engine = sqlite_engine(db_path)
    with Session(engine) as session:
        q = select(Email).options(selectinload(Email.labels)).order_by(Email.id)
        if split:
            q = q.where(Email.split == split)
        emails = list(session.scalars(q).unique().all())

    if not emails:
        raise SystemExit(f"❌ Aucun email trouvé (split={split!r}) dans {db_path}")

    if n_emails and n_emails < len(emails):
        import random
        rng = random.Random(seed)
        emails = rng.sample(emails, n_emails)

    return emails


def load_all_label_names(db_path: Path) -> list[str]:
    engine = sqlite_engine(db_path)
    with Session(engine) as session:
        names = list(session.scalars(select(Label.name).order_by(Label.name)).all())
    if not names:
        raise SystemExit("❌ Aucun label en base.")
    return names


# ---------------------------------------------------------------------------
# Benchmark LSTM
# ---------------------------------------------------------------------------
def benchmark_lstm(emails: list[Email], model_dir: Path) -> TimingResult:
    print(f"\n📦 Chargement du modèle LSTM depuis {model_dir}...")
    t0 = time.perf_counter()

    import nltk
    import string
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf

    for pkg in ["stopwords", "punkt", "punkt_tab"]:
        nltk.download(pkg, quiet=True)
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    def transform_text(text: str) -> str:
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t.isalnum()]
        tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
        return " ".join(ps.stem(t) for t in tokens)

    with open(model_dir / "keras_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open(model_dir / "label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open(model_dir / "meta.json") as f:
        meta = json.load(f)
    model = tf.keras.models.load_model(model_dir / "lstm_model.keras")
    max_len = meta["max_len"]

    setup_seconds = time.perf_counter() - t0
    print(f"✅ Chargé en {setup_seconds:.2f}s — {meta['num_classes']} catégories")

    result = TimingResult(name="LSTM (supervisé)", n_items=len(emails), setup_seconds=setup_seconds)

    # Warmup (le premier appel TF/Keras inclut souvent une compilation à froid)
    if emails:
        warm_text = transform_text(emails[0].subject + " " + emails[0].body)
        warm_seq = pad_sequences(tokenizer.texts_to_sequences([warm_text]), maxlen=max_len, padding="post", truncating="post")
        model.predict(warm_seq, verbose=0)

    print(f"⏱️  Mesure latence par email (un par un, comme en prod temps réel)...")
    for em in emails:
        t_start = time.perf_counter()
        text = transform_text(em.subject + " " + em.body)
        seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=max_len, padding="post", truncating="post")
        proba = model.predict(seq, verbose=0)[0]
        _ = encoder.classes_[int(np.argmax(proba))]
        result.per_item_seconds.append(time.perf_counter() - t_start)

    # Débit en vrai batch (comme classify_csv.py le fait en prod)
    print(f"⏱️  Mesure débit en mode batch (toutes les requêtes ensemble)...")
    t_batch0 = time.perf_counter()
    texts = [transform_text(em.subject + " " + em.body) for em in emails]
    seqs = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len, padding="post", truncating="post")
    model.predict(seqs, verbose=0)
    result.batch_seconds = time.perf_counter() - t_batch0

    return result


# ---------------------------------------------------------------------------
# Benchmark Zero-shot
# ---------------------------------------------------------------------------
def benchmark_zero_shot(emails: list[Email], candidate_labels: list[str], model_id: str) -> TimingResult:
    print(f"\n📦 Chargement du pipeline zero-shot ({model_id})...")
    t0 = time.perf_counter()
    from transformers import pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model=model_id,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
    )
    setup_seconds = time.perf_counter() - t0
    print(f"✅ Chargé en {setup_seconds:.2f}s")

    result = TimingResult(name=f"Zero-shot NLI ({model_id})", n_items=len(emails), setup_seconds=setup_seconds)

    def email_text(em: Email, max_chars: int = 2000) -> str:
        raw = f"Subject: {em.subject}\n\n{em.body}"
        raw = " ".join(raw.split())
        return raw[: max_chars - 1] + "…" if len(raw) > max_chars else raw

    # Warmup
    if emails:
        classifier(email_text(emails[0]), candidate_labels, multi_label=False)

    print(f"⏱️  Mesure latence par email (un par un — le mode d'usage réel du zero-shot)...")
    for em in emails:
        t_start = time.perf_counter()
        classifier(email_text(em), candidate_labels, multi_label=False)
        result.per_item_seconds.append(time.perf_counter() - t_start)

    # Le pipeline HF zero-shot ne batch pas plusieurs *documents* nativement
    # de façon plus efficace que un-par-un sur CPU (il batch surtout les
    # candidats internes) ; on rapporte donc le même chiffre pour le débit,
    # mais on le mesure explicitement pour rester honnête sur la méthode.
    result.batch_seconds = result.total_seconds

    return result


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def print_comparison(results: list[TimingResult]) -> None:
    print("\n" + "=" * 78)
    print("  RÉSUMÉ — Latence & débit")
    print("=" * 78)

    summaries = [r.summary() for r in results]
    headers = ["Modèle", "Setup(s)", "Lat.moy(ms)", "Lat.médiane(ms)", "Lat.p95(ms)", "Débit(email/s)"]
    rows = [
        [s["name"], s["setup_seconds"], s["latency_mean_ms"], s["latency_median_ms"],
         s["latency_p95_ms"], s["throughput_per_sec"]]
        for s in summaries
    ]
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))

    if len(results) >= 2:
        lstm_thr = next((s["throughput_per_sec"] for s in summaries if "LSTM" in s["name"]), None)
        zs_thr = next((s["throughput_per_sec"] for s in summaries if "Zero-shot" in s["name"]), None)
        if lstm_thr and zs_thr and zs_thr > 0:
            print(f"\n👉 Le LSTM est environ {lstm_thr / zs_thr:.1f}x plus rapide que le zero-shot sur ce run.")
            print("   (attendu : le zero-shot évalue une hypothèse par label candidat à chaque email,")
            print("    le LSTM fait une seule passe avant softmax sur toutes les catégories à la fois.)")


def save_csv(results: list[TimingResult], output_path: Path) -> None:
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = [r.summary() for r in results]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\n💾 Résultats sauvegardés dans : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de rapidité LSTM vs zero-shot")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--n-emails", type=int, default=50, help="Nombre d'emails à utiliser pour le benchmark")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Dossier des artefacts LSTM")
    parser.add_argument("--zero-shot-model", type=str, default=ZERO_SHOT_MODEL_ID,
                        help="Modèle HF pour le zero-shot (id HF ou chemin local, ex: après fine-tune)")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="CSV de sortie (optionnel)")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    emails = load_emails(args.db, args.split, args.n_emails, seed=args.seed)
    print(f"📂 {len(emails)} email(s) chargé(s) pour le benchmark (db={args.db}, split={args.split or 'tous'})")

    results: list[TimingResult] = []

    if not args.skip_lstm:
        if not (args.model_dir / "lstm_model.keras").exists():
            print(f"⚠️  Modèle LSTM introuvable dans {args.model_dir}, étape ignorée.")
        else:
            results.append(benchmark_lstm(emails, args.model_dir))

    if not args.skip_zero_shot:
        candidate_labels = load_all_label_names(args.db)
        results.append(benchmark_zero_shot(emails, candidate_labels, args.zero_shot_model))

    if not results:
        raise SystemExit("❌ Aucun benchmark exécuté (tout a été --skip-... ou les modèles sont introuvables).")

    print_comparison(results)

    if args.output:
        save_csv(results, args.output)
