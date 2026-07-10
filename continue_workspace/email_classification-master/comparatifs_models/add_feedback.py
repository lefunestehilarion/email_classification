#!/usr/bin/env python3
"""
add_feedback.py — Enregistre une correction utilisateur sur une prédiction zero-shot.

But : capturer "le modèle a prédit X, la vraie catégorie est Y" pour pouvoir
ensuite ré-entraîner (fine_tune_zero_shot.py) sans casser la capacité
zero-shot du modèle sur des catégories jamais vues.

Usage :
    # Une correction ponctuelle (CLI)
    python add_feedback.py --email-id 42 --predicted "Spam" --correct "Business" --score 0.61

    # Import en masse depuis un CSV (email_id,label_predit,label_correct,score_predit)
    python add_feedback.py --csv corrections.csv

    # Voir le feedback accumulé non encore utilisé pour l'entraînement
    python add_feedback.py --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import Base, Email, Feedback, sqlite_engine

DEFAULT_DB = Path(__file__).resolve().parent / "data" / "emails.db"


def ensure_feedback_table(db_path: Path) -> None:
    engine = sqlite_engine(db_path)
    Base.metadata.create_all(engine)  # no-op si les tables existent déjà


def add_one(db_path: Path, email_id: int, predicted: str, correct: str,
            score: float | None = None, model_id: str | None = None) -> None:
    engine = sqlite_engine(db_path)
    with Session(engine) as session:
        email = session.get(Email, email_id)
        if email is None:
            raise SystemExit(f"❌ Aucun email avec id={email_id} dans la base.")

        fb = Feedback(
            email_id=email_id,
            label_predit=predicted,
            label_correct=correct,
            score_predit=score,
            model_id=model_id,
        )
        session.add(fb)
        session.commit()
        print(f"✅ Feedback enregistré (id={fb.id}) : email {email_id} — "
              f"prédit={predicted!r} → correct={correct!r}")


def add_from_csv(db_path: Path, csv_path: Path) -> None:
    """CSV attendu : email_id,label_predit,label_correct[,score_predit][,model_id]"""
    df = pd.read_csv(csv_path)
    required = {"email_id", "label_predit", "label_correct"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"❌ Colonnes manquantes dans le CSV : {missing}")

    engine = sqlite_engine(db_path)
    n_ok, n_skip = 0, 0
    with Session(engine) as session:
        for _, row in df.iterrows():
            email_id = int(row["email_id"])
            email = session.get(Email, email_id)
            if email is None:
                print(f"  ⚠️  email_id={email_id} introuvable, ignoré")
                n_skip += 1
                continue
            fb = Feedback(
                email_id=email_id,
                label_predit=str(row["label_predit"]),
                label_correct=str(row["label_correct"]),
                score_predit=float(row["score_predit"]) if "score_predit" in df.columns and pd.notna(row.get("score_predit")) else None,
                model_id=str(row["model_id"]) if "model_id" in df.columns and pd.notna(row.get("model_id")) else None,
            )
            session.add(fb)
            n_ok += 1
        session.commit()

    print(f"✅ {n_ok} feedback(s) importé(s) | ⚠️ {n_skip} ignoré(s)")


def list_pending(db_path: Path) -> None:
    engine = sqlite_engine(db_path)
    with Session(engine) as session:
        rows = session.scalars(
            select(Feedback).where(Feedback.used_in_training == 0).order_by(Feedback.created_at)
        ).all()

    if not rows:
        print("Aucun feedback en attente d'utilisation pour l'entraînement.")
        return

    print(f"📋 {len(rows)} feedback(s) en attente :\n")
    for fb in rows:
        score_str = f"{fb.score_predit:.2%}" if fb.score_predit is not None else "?"
        print(f"  [{fb.id:>4}] email={fb.email_id:<6} "
              f"prédit={fb.label_predit!r:<25} → correct={fb.label_correct!r:<25} "
              f"(score={score_str}, créé={fb.created_at})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enregistre des corrections utilisateur sur le zero-shot")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--email-id", type=int)
    parser.add_argument("--predicted", type=str, help="Label prédit par le modèle")
    parser.add_argument("--correct", type=str, help="Vrai label (correction)")
    parser.add_argument("--score", type=float, default=None, help="Score de confiance de la prédiction d'origine")
    parser.add_argument("--model-id", type=str, default=None, help="Identifiant du modèle utilisé (ex: facebook/bart-large-mnli, ou chemin local après fine-tune)")
    parser.add_argument("--csv", type=Path, default=None, help="Import en masse depuis un CSV")
    parser.add_argument("--list", action="store_true", help="Lister le feedback non encore utilisé pour l'entraînement")
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    ensure_feedback_table(args.db)

    if args.list:
        list_pending(args.db)
    elif args.csv:
        add_from_csv(args.db, args.csv)
    elif args.email_id is not None and args.predicted and args.correct:
        add_one(args.db, args.email_id, args.predicted, args.correct, args.score, args.model_id)
    else:
        parser.print_help()
