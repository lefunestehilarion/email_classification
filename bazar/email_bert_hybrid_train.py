#!/usr/bin/env python3
"""
Entraînement classificateur e-mails : backbone Transformers gelé → [CLS] → 3 stats textuelles
(caractères, mots, phrases) → tête Keras (softmax, sparse_categorical_crossentropy).
CSV : colonnes subject, body, label. Voir README_bert_hybrid_uv.md pour l’usage avec uv.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model


MODELS_DICT = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "albert": "albert-base-v2",
    "tinybert": "huawei-noah/TinyBERT_General_6L_768D",
}


def ensure_nltk() -> None:
    for pkg in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def load_backbone(model_choice: str) -> tuple[AutoTokenizer, AutoModel, str]:
    name = MODELS_DICT.get(model_choice)
    if not name:
        raise ValueError(f"Modèle inconnu: {model_choice}. Choisir parmi {list(MODELS_DICT)}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return tokenizer, model, name


def build_frame(csv_path: Path) -> tuple[pd.DataFrame, LabelEncoder]:
    df = pd.read_csv(csv_path, encoding="utf-8")
    required = {"subject", "body", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes CSV requises: {sorted(required)} — manquantes: {sorted(missing)}")
    df = df.copy()
    df["text"] = (df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)).str.strip()
    df["num_characters"] = df["text"].apply(len)
    df["num_words"] = df["text"].apply(lambda x: len(word_tokenize(x)))
    df["num_sentence"] = df["text"].apply(lambda x: len(sent_tokenize(x)))
    encoder = LabelEncoder()
    df["target"] = encoder.fit_transform(df["label"])
    return df, encoder


def encode_ids(
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
) -> np.ndarray:
    enc = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return enc["input_ids"]


@torch.no_grad()
def bert_cls_matrix(
    bert: AutoModel,
    input_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    bert = bert.to(device)
    out_list: list[np.ndarray] = []
    n = len(input_ids)
    for start in range(0, n, batch_size):
        batch = torch.tensor(input_ids[start : start + batch_size], device=device)
        outputs = bert(batch)
        cls = outputs[0][:, 0, :].cpu().numpy()
        out_list.append(cls)
    return np.vstack(out_list)


def build_keras_head(hidden: int, num_classes: int) -> Model:
    input_bert = Input(shape=(hidden,), name="bert_features")
    input_stats_layer = Input(shape=(3,), name="input_stats")
    stats_feat = Dense(8, activation="relu")(input_stats_layer)
    combined = concatenate([input_bert, stats_feat])
    x = Dense(32, activation="relu")(combined)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation="softmax")(x)
    m = Model(inputs=[input_bert, input_stats_layer], outputs=output)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne le classificateur BERT + stats + Keras.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV avec colonnes subject, body, label")
    parser.add_argument(
        "--backbone",
        default="distilbert",
        choices=list(MODELS_DICT.keys()),
        help="Identifiant Hugging Face prédéfini (distilbert par défaut).",
    )
    parser.add_argument("--max-length", type=int, default=50, help="Longueur max des IDs tokenisés.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction de l’ensemble de test.")
    parser.add_argument("--random-state", type=int, default=2, help="Graine du train_test_split.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bert-batch", type=int, default=32, help="Taille des lots pour l'inférence BERT (mémoire).")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts_bert_hybrid"), help="Dossier de sortie.")
    parser.add_argument("--cpu-only", action="store_true", help="Forcer le calcul BERT sur CPU.")
    args = parser.parse_args()

    ensure_nltk()
    device = torch.device("cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda")

    df, encoder = build_frame(args.csv)
    tokenizer, bert, hf_name = load_backbone(args.backbone)
    hidden = int(bert.config.hidden_size)

    print(f"Backbone: {hf_name} | hidden_size={hidden} | device={device}")

    texts = df["text"].tolist()
    X_ids = encode_ids(tokenizer, texts, args.max_length)
    X_stats = df[["num_characters", "num_words", "num_sentence"]].values.astype(np.float32)
    y = df["target"].values

    train_ids, test_ids, train_stats, test_stats, y_train, y_test = train_test_split(
        X_ids,
        X_stats,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("Pré-calcul des embeddings [CLS] (peut prendre plusieurs minutes)...")
    train_cls = bert_cls_matrix(bert, train_ids, args.bert_batch, device)
    test_cls = bert_cls_matrix(bert, test_ids, args.bert_batch, device)
    print(f"train_cls {train_cls.shape} | test_cls {test_cls.shape}")

    num_classes = len(encoder.classes_)
    model = build_keras_head(hidden, num_classes)
    print(f"Paramètres Keras: {model.count_params():,}")

    model.fit(
        x=[train_cls, train_stats],
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=([test_cls, test_stats], y_test),
        verbose=1,
    )

    y_prob = model.predict([test_cls, test_stats], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    print(classification_report(y_test, y_pred, target_names=list(encoder.classes_), zero_division=0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.out_dir / "keras_classifier.keras")
    with open(args.out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    meta = {
        "hf_model": hf_name,
        "backbone_key": args.backbone,
        "max_length": args.max_length,
        "classes": list(encoder.classes_),
        "label_mapping": {str(c): int(i) for i, c in enumerate(encoder.classes_)},
    }
    with open(args.out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Artefacts écrits dans {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
