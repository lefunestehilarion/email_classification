#!/usr/bin/env python3
"""
fine_tune_lora.py — Fine-tuning bon marché du zero-shot NLI (LoRA/PEFT) sur le
feedback.csv, sans dépendre de SQLite.

Pourquoi LoRA plutôt qu'un fine-tune complet (comme fine_tune_zero_shot.py) :
    bart-large-mnli a ~407M paramètres. Un fine-tune complet réentraîne TOUS
    ces poids : lent, gourmand en mémoire (états d'optimiseur sur 400M+
    params), et risque d'abîmer davantage la capacité zero-shot générale sur
    des catégories jamais vues.
    LoRA gèle le modèle de base et n'entraîne que de petites matrices
    "adapter" insérées dans les projections d'attention (q_proj, v_proj) :
    typiquement <1 % des paramètres totaux. Résultat concret :
      - training beaucoup plus rapide, même sur peu de feedback
      - l'adapter sauvegardé pèse quelques Mo (au lieu de ~1.6 Go pour le
        modèle entier)
      - le modèle de base reste intact : tu peux garder plusieurs adapters
        (un par domaine/version) et les charger/décharger à la volée

Installation :
    pip install peft --break-system-packages

Usage :
    python fine_tune_lora.py --feedback-csv feedback.csv --output-dir models/bart-mnli-lora

    # Reprendre l'entraînement sur un adapter existant
    python fine_tune_lora.py --feedback-csv feedback.csv --output-dir models/bart-mnli-lora-v2 \\
        --base-adapter models/bart-mnli-lora/final
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_MODEL_ID = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This email is about {}."
LABEL_CONTRADICTION, LABEL_NEUTRAL, LABEL_ENTAILMENT = 0, 1, 2


@dataclass
class NLIExample:
    premise: str
    hypothesis: str
    nli_label: int


def email_text(subject, body, max_chars: int = 2000) -> str:
    raw = f"Subject: {subject}\n\n{body}"
    raw = " ".join(str(raw).split())
    return raw if len(raw) <= max_chars else raw[: max_chars - 1] + "…"


def build_examples(
    fdf: pd.DataFrame, all_labels: list[str], n_random_negatives: int, max_chars: int, seed: int,
) -> tuple[list[NLIExample], list[int]]:
    """Même logique que fine_tune_zero_shot.py : 1 positif (entailment) + 1
    négatif dur (l'erreur du modèle, contradiction) + n négatifs aléatoires,
    pour garder la discrimination entre les autres catégories."""
    rng = random.Random(seed)
    pending = fdf[fdf["used_in_training"] == 0]
    examples: list[NLIExample] = []
    used_idx: list[int] = []

    for idx, fb in pending.iterrows():
        text = email_text(fb["subject"], fb["body"], max_chars=max_chars)
        examples.append(NLIExample(text, HYPOTHESIS_TEMPLATE.format(fb["correct_label"]), LABEL_ENTAILMENT))

        predicted = fb.get("predicted_label", "")
        if isinstance(predicted, str) and predicted and predicted != fb["correct_label"]:
            examples.append(NLIExample(text, HYPOTHESIS_TEMPLATE.format(predicted), LABEL_CONTRADICTION))

        pool = [l for l in all_labels if l not in (fb["correct_label"], predicted)]
        rng.shuffle(pool)
        for neg in pool[:n_random_negatives]:
            examples.append(NLIExample(text, HYPOTHESIS_TEMPLATE.format(neg), LABEL_CONTRADICTION))

        used_idx.append(idx)

    return examples, used_idx


class NLIDataset(Dataset):
    def __init__(self, examples: list[NLIExample], tokenizer, max_length: int = 512):
        self.examples, self.tokenizer, self.max_length = examples, tokenizer, max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        ex = self.examples[i]
        enc = self.tokenizer(ex.premise, ex.hypothesis, truncation=True, max_length=self.max_length)
        enc["labels"] = ex.nli_label
        return enc


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LoRA (bon marché) du zero-shot NLI sur feedback.csv")
    parser.add_argument("--feedback-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_ID)
    parser.add_argument("--base-adapter", type=str, default=None, help="Reprendre depuis un adapter LoRA existant")
    parser.add_argument("--epochs", type=int, default=5, help="LoRA tolère plus d'époques sans trop surapprendre")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA tolère un LR plus élevé qu'un full fine-tune")
    parser.add_argument("--lora-r", type=int, default=8, help="Rang des matrices LoRA (8-16 suffit en général)")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--n-random-negatives", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--no-mark-used", action="store_true")
    args = parser.parse_args()

    fdf = pd.read_csv(args.feedback_csv)
    if fdf.empty or (fdf["used_in_training"] == 0).sum() == 0:
        print("⚠️  Aucun feedback non utilisé trouvé. Rien à entraîner.")
        print("    → ajoute du feedback avec add_feedback_csv.py d'abord.")
        return

    all_labels = sorted(set(fdf["correct_label"]) | set(fdf["predicted_label"].dropna()))
    examples, used_idx = build_examples(fdf, all_labels, args.n_random_negatives, args.max_chars, args.seed)
    print(f"📋 {len(used_idx)} feedback(s) source -> {len(examples)} exemples NLI générés")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_ratio)) if len(examples) >= 5 else 0
    val_examples, train_examples = examples[:n_val], examples[n_val:]
    print(f"   train={len(train_examples)} | val={len(val_examples)}")

    print(f"📦 Chargement du modèle de base : {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)

    if args.base_adapter:
        print(f"📦 Reprise depuis l'adapter : {args.base_adapter}")
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # attention de BART, suffisant en pratique
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()  # confirme visuellement le <1 % de params entraînés

    train_ds = NLIDataset(train_examples, tokenizer)
    val_ds = NLIDataset(val_examples, tokenizer) if val_examples else None
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss" if val_ds else None,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator,
    )
    print(f"🚀 Fine-tuning LoRA ({args.epochs} époques, lr={args.lr}, r={args.lora_r})...")
    trainer.train()

    final_path = args.output_dir / "final"
    model.save_pretrained(final_path)  # ne sauvegarde QUE l'adapter (quelques Mo)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Adapter LoRA sauvegardé dans : {final_path}")
    print("   → pour reclassifier avec cet adapter, charge le modèle de base puis :")
    print(f"     PeftModel.from_pretrained(model, '{final_path}')")
    print("     (zero_shot_csv.py charge un modèle complet par défaut ; dis-moi si tu veux")
    print("      que je l'adapte pour charger un adapter LoRA directement via --model)")

    if not args.no_mark_used:
        fdf.loc[used_idx, "used_in_training"] = 1
        fdf.to_csv(args.feedback_csv, index=False)
        print(f"📝 {len(used_idx)} feedback(s) marqué(s) comme utilisés")
    else:
        print("ℹ️  Feedback NON marqué comme utilisé (--no-mark-used)")


if __name__ == "__main__":
    main()
