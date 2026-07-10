#!/usr/bin/env python3
"""
fine_tune_zero_shot.py — Fine-tune le modèle NLI zero-shot (bart-large-mnli)
sur les corrections utilisateur stockées dans la table `feedback`.

Principe :
    Le zero-shot classifie en testant, pour chaque label candidat, si l'email
    "implique" (entailment) la phrase "This email is about {label}.". Plutôt
    que de remplacer ce mécanisme par une tête de classification classique
    (ce qui ferait perdre la capacité à gérer une catégorie jamais vue), on
    continue d'entraîner le modèle sur la tâche NLI elle-même : on lui donne
    des exemples (email, hypothèse, label_NLI) où :
        - hypothèse construite avec le BON label  -> "entailment"
        - hypothèse construite avec un MAUVAIS label (souvent celui que le
          modèle a prédit à tort) -> "contradiction"

    Le modèle reste donc un classifieur NLI généraliste, juste mieux calibré
    sur ton domaine (vocabulaire métier, ambiguïtés fréquentes entre tes
    catégories). Il garde sa capacité zero-shot pour une catégorie nouvelle
    qu'il n'a jamais vue dans le feedback.

Usage :
    # Fine-tuner sur tout le feedback non encore utilisé
    python fine_tune_zero_shot.py --db data/emails.db --output-dir models/bart-mnli-finetuned

    # Inclure aussi des négatifs aléatoires (labels au hasard) en plus des
    # erreurs réelles, pour mieux couvrir l'espace des labels
    python fine_tune_zero_shot.py --db data/emails.db --output-dir models/bart-mnli-finetuned --n-random-negatives 2

    # Reprendre l'entraînement à partir d'un précédent fine-tune (au lieu du modèle de base)
    python fine_tune_zero_shot.py --db data/emails.db --output-dir models/bart-mnli-finetuned-v2 --base-model models/bart-mnli-finetuned
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from models import Email, Feedback, Label, sqlite_engine

DEFAULT_DB = Path(__file__).resolve().parent / "data" / "emails.db"
BASE_MODEL_ID = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This email is about {}."

# Convention MNLI standard pour bart-large-mnli : 0=contradiction, 1=neutral, 2=entailment
LABEL_CONTRADICTION = 0
LABEL_NEUTRAL = 1
LABEL_ENTAILMENT = 2


@dataclass
class NLIExample:
    premise: str
    hypothesis: str
    nli_label: int


def email_text(subject: str, body: str, max_chars: int = 2000) -> str:
    raw = f"Subject: {subject}\n\n{body}"
    raw = " ".join(raw.split())
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 1] + "…"


def load_all_label_names(session: Session) -> list[str]:
    return list(session.scalars(select(Label.name).order_by(Label.name)).all())


def build_examples_from_feedback(
    session: Session,
    all_labels: list[str],
    n_random_negatives: int = 1,
    max_chars: int = 2000,
    seed: int = 2,
) -> list[NLIExample]:
    """
    Pour chaque ligne de feedback non encore utilisée :
      - 1 exemple positif : hypothèse avec label_correct -> entailment
      - 1 exemple négatif "dur" : hypothèse avec label_predit (l'erreur du
        modèle) -> contradiction, SAUF si label_predit == label_correct
        (cas où le feedback confirme une bonne prédiction : on ne crée pas
        de négatif inutile)
      - n_random_negatives exemples négatifs aléatoires parmi les autres
        labels, pour éviter que le modèle ne fasse que mémoriser la confusion
        signalée et continue de bien discriminer les autres catégories.
    """
    rng = random.Random(seed)
    rows = session.scalars(
        select(Feedback).where(Feedback.used_in_training == 0)
    ).all()

    examples: list[NLIExample] = []
    feedback_ids_used: list[int] = []

    for fb in rows:
        email = session.get(Email, fb.email_id)
        if email is None:
            continue
        text = email_text(email.subject, email.body, max_chars=max_chars)

        # Positif : le vrai label
        examples.append(NLIExample(
            premise=text,
            hypothesis=HYPOTHESIS_TEMPLATE.format(fb.label_correct),
            nli_label=LABEL_ENTAILMENT,
        ))

        # Négatif dur : ce que le modèle avait prédit à tort
        if fb.label_predit and fb.label_predit != fb.label_correct:
            examples.append(NLIExample(
                premise=text,
                hypothesis=HYPOTHESIS_TEMPLATE.format(fb.label_predit),
                nli_label=LABEL_CONTRADICTION,
            ))

        # Négatifs aléatoires supplémentaires (autres labels, hors bon label et hors label prédit déjà couvert)
        pool = [l for l in all_labels if l not in (fb.label_correct, fb.label_predit)]
        rng.shuffle(pool)
        for neg_label in pool[:n_random_negatives]:
            examples.append(NLIExample(
                premise=text,
                hypothesis=HYPOTHESIS_TEMPLATE.format(neg_label),
                nli_label=LABEL_CONTRADICTION,
            ))

        feedback_ids_used.append(fb.id)

    return examples, feedback_ids_used


class NLIDataset(Dataset):
    def __init__(self, examples: list[NLIExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.premise,
            ex.hypothesis,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = ex.nli_label
        return enc


def mark_feedback_used(db_path: Path, feedback_ids: list[int]) -> None:
    if not feedback_ids:
        return
    engine = sqlite_engine(db_path)
    with Session(engine) as session:
        session.execute(
            update(Feedback).where(Feedback.id.in_(feedback_ids)).values(used_in_training=1)
        )
        session.commit()


def fine_tune(
    db_path: Path,
    output_dir: Path,
    base_model: str = BASE_MODEL_ID,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    n_random_negatives: int = 1,
    max_chars: int = 2000,
    val_ratio: float = 0.15,
    seed: int = 2,
    mark_used: bool = True,
) -> None:
    engine = sqlite_engine(db_path)
    with Session(engine) as session:
        all_labels = load_all_label_names(session)
        if not all_labels:
            raise SystemExit("❌ Aucun label en base — rien à utiliser comme négatifs aléatoires.")

        examples, feedback_ids = build_examples_from_feedback(
            session, all_labels, n_random_negatives=n_random_negatives,
            max_chars=max_chars, seed=seed,
        )

    if not examples:
        print("⚠️  Aucun feedback non utilisé trouvé. Rien à entraîner.")
        print("    → ajoute du feedback avec add_feedback.py d'abord.")
        return

    print(f"📋 {len(feedback_ids)} feedback(s) source -> {len(examples)} exemples NLI générés")
    print(f"   (1 positif + négatif(s) dur/aléatoires par feedback)")

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_ratio)) if len(examples) >= 5 else 0
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    print(f"   train={len(train_examples)} | val={len(val_examples)}")

    print(f"📦 Chargement du modèle de base : {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)

    train_ds = NLIDataset(train_examples, tokenizer)
    val_ds = NLIDataset(val_examples, tokenizer) if val_examples else None

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss" if val_ds else None,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print(f"🚀 Fine-tuning ({epochs} époques, lr={learning_rate})...")
    trainer.train()

    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Modèle fine-tuné sauvegardé dans : {final_path}")
    print(f"   → utilise ce chemin comme MODEL_ID dans test_mdeberta_zero_shot.py")
    print(f"     (ou via --model-dir si tu adaptes le script pour l'accepter en argument)")

    if mark_used:
        mark_feedback_used(db_path, feedback_ids)
        print(f"📝 {len(feedback_ids)} feedback(s) marqué(s) comme utilisés (used_in_training=1)")
    else:
        print("ℹ️  Feedback NON marqué comme utilisé (--no-mark-used) : il sera réutilisé au prochain run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune le modèle NLI zero-shot sur le feedback utilisateur")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_ID,
                        help="Modèle de départ : facebook/bart-large-mnli, ou un chemin local vers un fine-tune précédent")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n-random-negatives", type=int, default=1,
                        help="Nb de labels aléatoires utilisés comme négatifs supplémentaires par exemple de feedback")
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--no-mark-used", action="store_true",
                        help="Ne pas marquer le feedback comme utilisé (pour itérer/tester sans le consommer)")
    args = parser.parse_args()

    fine_tune(
        db_path=args.db,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_random_negatives=args.n_random_negatives,
        max_chars=args.max_chars,
        val_ratio=args.val_ratio,
        seed=args.seed,
        mark_used=not args.no_mark_used,
    )
