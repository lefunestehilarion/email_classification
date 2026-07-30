#!/usr/bin/env python3
"""
zero_shot_csv.py — Classifie des mails (CSV) avec le zero-shot NLI, sur des
catégories saisies dynamiquement (CLI ou prompt interactif), sans dépendre
d'une base SQLite.

CSV d'entrée attendu : colonnes "subject" et "body" (un "id" optionnel,
sinon l'index de ligne sert d'identifiant).

Usage :
    # Catégories passées en argument
    python zero_shot_csv.py --input mails.csv --categories "Spam,Business,Support,Facture"

    # Catégories saisies au prompt (si --categories absent)
    python zero_shot_csv.py --input mails.csv

    # Utiliser un modèle fine-tuné (full fine-tune) au lieu du modèle de base
    python zero_shot_csv.py --input mails.csv --model models/bart-mnli-finetuned/final
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import pipeline

DEFAULT_MODEL = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This email is about {}."


def load_categories(args: argparse.Namespace) -> list[str]:
    if args.categories:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif args.categories_file:
        cats = [
            line.strip()
            for line in Path(args.categories_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw = input("Catégories (séparées par des virgules) : ")
        cats = [c.strip() for c in raw.split(",") if c.strip()]
    if len(cats) < 2:
        raise SystemExit("❌ Il faut au moins 2 catégories.")
    return cats


def email_text(subject, body, max_chars: int = 4000) -> str:
    raw = f"Subject: {subject}\n\n{body}"
    raw = " ".join(str(raw).split())
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tri zero-shot de mails CSV, catégories dynamiques")
    parser.add_argument("--input", type=Path, required=True, help="CSV d'entrée (colonnes subject, body[, id])")
    parser.add_argument("--output", type=Path, default=None, help="CSV de sortie (défaut: <input>_classified.csv)")
    parser.add_argument("--categories", type=str, default=None, help="Liste séparée par virgules")
    parser.add_argument("--categories-file", type=Path, default=None, help="Fichier avec une catégorie par ligne")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Modèle HF ou chemin local (fine-tuné)")
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument(
        "--multi-label", action="store_true",
        help="Autoriser plusieurs labels indépendants (sinon softmax exclusif entre catégories)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "subject" not in df.columns or "body" not in df.columns:
        raise SystemExit("❌ Le CSV doit contenir les colonnes 'subject' et 'body'.")
    if "id" not in df.columns:
        df.insert(0, "id", df.index)

    categories = load_categories(args)
    print(f"📋 Catégories ({len(categories)}) : {', '.join(categories)}")

    print(f"📦 Chargement du modèle : {args.model}")
    classifier = pipeline(
        "zero-shot-classification",
        model=args.model,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
    )
    print("Pipeline prêt.\n")

    preds, scores = [], []
    for _, row in df.iterrows():
        text = email_text(row["subject"], row["body"], max_chars=args.max_chars)
        out = classifier(text, categories, multi_label=args.multi_label)
        preds.append(out["labels"][0])
        scores.append(float(out["scores"][0]))
        print(f"  id={row['id']}  →  {out['labels'][0]}  ({out['scores'][0] * 100:.1f} %)")

    df["predicted_label"] = preds
    df["predicted_score"] = scores

    output = args.output or args.input.with_name(args.input.stem + "_classified.csv")
    df.to_csv(output, index=False)
    print(f"\n✅ Résultats écrits dans : {output}")
    print(
        "   → pour corriger une prédiction : "
        "python add_feedback_csv.py --classified " + str(output) + " --email-id <id> --correct <Label>"
    )


if __name__ == "__main__":
    main()
