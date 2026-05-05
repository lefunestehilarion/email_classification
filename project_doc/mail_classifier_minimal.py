import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


ALLOWED_LABELS = {"pro", "sport", "ecole"}


def clean_text(text: str) -> str:
    """Texte en caractères latins « plats » (dataset et inférence supposés en anglais)."""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_dataset(csv_path: str, *, english_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    required_columns = {"subject", "body", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    df = df[df["label"].isin(ALLOWED_LABELS)].copy()
    raw_text = (
        df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    )
    df["text"] = raw_text.map(clean_text)

    if english_only:
        try:
            from langdetect import LangDetectException, detect
        except ImportError:
            print(
                "Avertissement: `langdetect` absent — filtre anglais désactivé. "
                "Installez avec: pip install langdetect"
            )
        else:

            def is_english(t: str) -> bool:
                if not (t or "").strip():
                    return False
                try:
                    return detect(t[:5000]) == "en"
                except LangDetectException:
                    return False

            mask = raw_text.map(is_english)
            before = len(df)
            df = df.loc[mask].copy()
            print(
                f"Filtre anglais: {len(df)} lignes conservées sur {before} "
                f"({before - len(df)} exclues)."
            )

    return df[["text", "label"]]


def train_model(csv_path: str, *, english_only: bool = True):
    df = build_dataset(csv_path, english_only=english_only)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    return model


if __name__ == "__main__":
    print(
        "Rappel: le rapport ci-dessous mesure la généralisation sur des mails "
        "du même CSV (même langue / même style). Des phrases en français sont "
        "hors distribution → prédictions peu fiables.\n"
    )
    model = train_model("dataset_emails_classification_2k_harder.csv")

    # Exemples en anglais (alignés avec l’entraînement du CSV).
    samples = [
        "Client meeting tomorrow — please review budget and timeline",
        "Practice tonight at the stadium, bring your cleats",
        "Homework due Monday for math class — submit on the portal",
    ]
    cleaned = [clean_text(s) for s in samples]
    predictions = model.predict(cleaned)

    for raw, label in zip(samples, predictions):
        print(f"{label:>6} <- {raw}")
