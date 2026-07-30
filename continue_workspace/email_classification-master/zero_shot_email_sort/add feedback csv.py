#!/usr/bin/env python3
"""
add_feedback_csv.py — Enregistre des corrections utilisateur sur des prédictions
zero-shot, en s'appuyant uniquement sur des CSV (pas de SQLite).

Le fichier de feedback est auto-suffisant : il stocke le texte du mail
directement (subject + body), pas juste un id, pour pouvoir fine-tuner sans
redépendre du CSV classifié d'origine.

Usage :
    # Correction unique
    python add_feedback_csv.py --classified mails_classified.csv --email-id 12 --correct "Facture"

    # Import en masse : CSV avec colonnes id,correct (le reste vient du classified)
    python add_feedback_csv.py --classified mails_classified.csv --corrections corrections.csv

    # Voir le feedback accumulé non encore utilisé pour l'entraînement
    python add_feedback_csv.py --list
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_FEEDBACK = Path("feedback.csv")
FEEDBACK_COLUMNS = [
    "id", "subject", "body", "predicted_label", "correct_label",
    "predicted_score", "used_in_training",
]


def load_feedback(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def save_feedback(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def add_one(classified: Path, feedback_path: Path, email_id: str, correct: str) -> None:
    cdf = pd.read_csv(classified)
    match = cdf[cdf["id"].astype(str) == str(email_id)]
    if match.empty:
        raise SystemExit(f"❌ id={email_id} introuvable dans {classified}")
    row = match.iloc[0]

    fdf = load_feedback(feedback_path)
    new_row = {
        "id": row["id"],
        "subject": row["subject"],
        "body": row["body"],
        "predicted_label": row.get("predicted_label", ""),
        "correct_label": correct,
        "predicted_score": row.get("predicted_score", None),
        "used_in_training": 0,
    }
    fdf = pd.concat([fdf, pd.DataFrame([new_row])], ignore_index=True)
    save_feedback(feedback_path, fdf)
    print(f"✅ Feedback enregistré : id={row['id']} prédit={row.get('predicted_label')!r} → correct={correct!r}")


def add_bulk(classified: Path, feedback_path: Path, corrections: Path) -> None:
    cdf = pd.read_csv(classified)
    corr = pd.read_csv(corrections)
    if "id" not in corr.columns or "correct" not in corr.columns:
        raise SystemExit("❌ Le CSV de corrections doit avoir les colonnes 'id' et 'correct'.")

    fdf = load_feedback(feedback_path)
    new_rows = []
    n_ok, n_skip = 0, 0
    for _, r in corr.iterrows():
        match = cdf[cdf["id"].astype(str) == str(r["id"])]
        if match.empty:
            n_skip += 1
            continue
        m = match.iloc[0]
        new_rows.append({
            "id": m["id"],
            "subject": m["subject"],
            "body": m["body"],
            "predicted_label": m.get("predicted_label", ""),
            "correct_label": r["correct"],
            "predicted_score": m.get("predicted_score", None),
            "used_in_training": 0,
        })
        n_ok += 1
    fdf = pd.concat([fdf, pd.DataFrame(new_rows)], ignore_index=True)
    save_feedback(feedback_path, fdf)
    print(f"✅ {n_ok} feedback(s) importé(s) | ⚠️ {n_skip} ignoré(s)")


def list_pending(feedback_path: Path) -> None:
    fdf = load_feedback(feedback_path)
    pending = fdf[fdf["used_in_training"] == 0]
    if pending.empty:
        print("Aucun feedback en attente d'utilisation pour l'entraînement.")
        return
    print(f"📋 {len(pending)} feedback(s) en attente :\n")
    for _, r in pending.iterrows():
        print(
            f"  id={r['id']}  prédit={r['predicted_label']!r} → correct={r['correct_label']!r} "
            f"(score={r['predicted_score']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feedback CSV pour zero-shot (sans SQLite)")
    parser.add_argument("--feedback-csv", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--classified", type=Path, help="CSV produit par zero_shot_csv.py")
    parser.add_argument("--email-id", type=str)
    parser.add_argument("--correct", type=str)
    parser.add_argument("--corrections", type=Path, help="CSV en masse (colonnes id,correct)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_pending(args.feedback_csv)
    elif args.corrections and args.classified:
        add_bulk(args.classified, args.feedback_csv, args.corrections)
    elif args.email_id is not None and args.correct and args.classified:
        add_one(args.classified, args.feedback_csv, args.email_id, args.correct)
    else:
        parser.print_help()
