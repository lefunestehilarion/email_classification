#!/usr/bin/env python3
"""
train.py — Entraînement du classifieur d'emails (10+ catégories)
Usage :
    python train.py --csv data/emails.csv
    python train.py --csv data/emails.csv --epochs 15 --max-features 5000
"""

import argparse
import os
import pickle
import json
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model

# ---------------------------------------------------------------------------
# NLTK data
# ---------------------------------------------------------------------------
for pkg in ['stopwords', 'punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

ps = PorterStemmer()


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(csv_path: str, output_dir: str = "models", epochs: int = 15,
          max_features: int = 3000, test_size: float = 0.2, random_state: int = 2):

    os.makedirs(output_dir, exist_ok=True)

    # --- Load & clean ---
    print(f"📂 Chargement de {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8')
    df = df[['subject', 'body', 'label']].copy()
    df['text'] = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).str.strip()
    df = df.drop_duplicates(keep='first').reset_index(drop=True)

    categories = sorted(df['label'].unique().tolist())
    print(f"✅ {len(df)} emails | {len(categories)} catégories : {categories}")

    # --- Encode labels ---
    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['label'])
    num_classes = len(encoder.classes_)

    # --- Preprocess text ---
    print("🔄 Préprocessing du texte...")
    df['transformed_text'] = df['text'].apply(transform_text)

    # --- TF-IDF features ---
    tfid = TfidfVectorizer(max_features=max_features)
    X_tfidf = tfid.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_tfidf, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- LSTM tokenization ---
    keras_tokenizer = Tokenizer(num_words=max_features)
    keras_tokenizer.fit_on_texts(df['transformed_text'])
    sequences = keras_tokenizer.texts_to_sequences(df['transformed_text'])
    lengths = [len(s) for s in sequences]
    max_len = int(np.percentile(lengths, 95))
    print(f"📏 Longueur max séquence (95e pct) : {max_len}")

    X_deep = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
        X_deep, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- Build LSTM model ---
    print("🏗️  Construction du modèle LSTM...")
    inp = Input(shape=(max_len,), name='text_input')
    x = Embedding(input_dim=max_features, output_dim=64)(inp)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # --- Train ---
    print(f"🚀 Entraînement ({epochs} époques)...")
    model.fit(
        X_tr_d, y_tr_d,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_te_d, y_te_d),
        verbose=1
    )

    # --- Evaluate ---
    y_pred = np.argmax(model.predict(X_te_d), axis=1)
    acc  = accuracy_score(y_te_d, y_pred)
    prec = precision_score(y_te_d, y_pred, average='macro', zero_division=0)
    print(f"\n📊 Accuracy : {acc:.4f} | Macro-Precision : {prec:.4f}")
    print(classification_report(y_te_d, y_pred,
                                 target_names=list(encoder.classes_),
                                 zero_division=0))

    # --- Save artifacts ---
    model.save(os.path.join(output_dir, "lstm_model.keras"))
    with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfid, f)
    with open(os.path.join(output_dir, "keras_tokenizer.pkl"), "wb") as f:
        pickle.dump(keras_tokenizer, f)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    meta = {
        "categories": categories,
        "num_classes": num_classes,
        "max_len": max_len,
        "max_features": max_features,
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4)
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Modèle et artefacts sauvegardés dans '{output_dir}/'")
    return meta


# ---------------------------------------------------------------------------
# Few-shot extension : ajouter des catégories sans réentraîner
# ---------------------------------------------------------------------------
def extend_categories(new_csv: str, output_dir: str = "models",
                       extra_epochs: int = 5, random_state: int = 2):
    """
    Ajoute de nouvelles catégories au modèle existant par fine-tuning partiel.
    Le CSV doit contenir uniquement les emails des NOUVELLES catégories.
    """
    print("🔄 Chargement des artefacts existants...")
    with open(os.path.join(output_dir, "keras_tokenizer.pkl"), "rb") as f:
        keras_tokenizer = pickle.load(f)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(output_dir, "meta.json")) as f:
        meta = json.load(f)

    model = tf.keras.models.load_model(os.path.join(output_dir, "lstm_model.keras"))
    max_len = meta["max_len"]
    existing_cats = meta["categories"]

    # --- Charger nouvelles données ---
    df_new = pd.read_csv(new_csv, encoding='utf-8')
    df_new = df_new[['subject', 'body', 'label']].copy()
    df_new['text'] = (df_new['subject'].fillna('') + ' ' + df_new['body'].fillna('')).str.strip()

    new_cats = [c for c in df_new['label'].unique() if c not in existing_cats]
    if not new_cats:
        print("⚠️  Aucune nouvelle catégorie détectée. Rien à faire.")
        return

    print(f"➕ Nouvelles catégories détectées : {new_cats}")
    all_cats = sorted(existing_cats + new_cats)

    # --- Réencoder les labels avec toutes les catégories ---
    new_encoder = LabelEncoder()
    new_encoder.fit(all_cats)
    df_new['target'] = new_encoder.transform(df_new['label'])
    num_classes_new = len(all_cats)

    # --- Préprocessing ---
    df_new['transformed_text'] = df_new['text'].apply(transform_text)
    sequences = keras_tokenizer.texts_to_sequences(df_new['transformed_text'])
    X_new = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y_new = df_new['target'].values

    # --- Reconstruire le modèle avec plus de sorties ---
    old_input   = model.input
    old_dense   = model.layers[-2].output  # Dense(64, relu)
    new_output  = Dense(num_classes_new, activation='softmax', name='new_output')(old_dense)
    new_model   = Model(inputs=old_input, outputs=new_output)

    # Geler les couches LSTM/Embedding pour ne réentraîner que la tête
    for layer in new_model.layers[:-2]:
        layer.trainable = False

    new_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    print(f"🚀 Fine-tuning ({extra_epochs} époques) sur {len(df_new)} emails...")
    new_model.fit(X_new, y_new, epochs=extra_epochs, batch_size=16, verbose=1)

    # --- Sauvegarder ---
    new_model.save(os.path.join(output_dir, "lstm_model.keras"))
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(new_encoder, f)

    meta["categories"] = all_cats
    meta["num_classes"] = num_classes_new
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Modèle étendu à {num_classes_new} catégories : {all_cats}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîneur de classifieur d'emails")
    subparsers = parser.add_subparsers(dest="command")

    # -- train --
    p_train = subparsers.add_parser("train", help="Entraîner depuis zéro")
    p_train.add_argument("--csv",          required=True, help="Chemin vers le CSV")
    p_train.add_argument("--output-dir",   default="models")
    p_train.add_argument("--epochs",       type=int, default=15)
    p_train.add_argument("--max-features", type=int, default=3000)
    p_train.add_argument("--test-size",    type=float, default=0.2)

    # -- extend --
    p_ext = subparsers.add_parser("extend", help="Ajouter des catégories")
    p_ext.add_argument("--csv",          required=True, help="CSV avec les nouvelles catégories")
    p_ext.add_argument("--output-dir",   default="models")
    p_ext.add_argument("--extra-epochs", type=int, default=5)

    args = parser.parse_args()

    if args.command == "train":
        train(args.csv, args.output_dir, args.epochs, args.max_features, args.test_size)
    elif args.command == "extend":
        extend_categories(args.csv, args.output_dir, args.extra_epochs)
    else:
        parser.print_help()
