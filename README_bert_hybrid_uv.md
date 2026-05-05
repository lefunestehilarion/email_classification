# Classifieur e-mails (BERT + stats + Keras)

Pipeline d’entraînement : **Transformers** (backbone gelé, défaut DistilBERT) → vecteur **`[CLS]`** → concaténation avec **3 nombres** dérivés du texte (caractères, mots, phrases) → **tête Keras** (`softmax`). Les classes viennent de la colonne `label` du CSV (autant de classes que de valeurs distinctes).

Script : `email_bert_hybrid_train.py`.

## Prérequis

- [uv](https://docs.astral.sh/uv/)
- CSV UTF-8 avec **`subject`**, **`body`**, **`label`**

## Installation et lancement

```bash
cd /chemin/vers/le/projet   # répertoire contenant pyproject.toml
uv sync
uv run python email_bert_hybrid_train.py --csv chemin/vers/data.csv --out-dir ./artifacts_bert_hybrid
```

Premier lancement : téléchargements lourds (PyTorch, TensorFlow, Hugging Face, données NLTK si besoin). Optionnel : `HF_TOKEN` si le hub Hugging Face limite les requêtes.

## Options principales

`--backbone` : `distilbert` (défaut), `bert`, `albert`, `tinybert`  
`--max-length` : longueur max des tokens (défaut `50`)  
`--test-size`, `--random-state` : découpage train / test  
`--epochs`, `--batch-size` : entraînement Keras  
`--bert-batch` : taille des lots pour le forward BERT (baisser en cas de manque de mémoire GPU)  
`--cpu-only` : BERT sur CPU uniquement

`--help` liste tout le détail.

## Fichiers produits (`--out-dir`)

- `keras_classifier.keras` — tête de classification entraînée  
- `label_encoder.pkl` — correspondance labels ↔ entiers  
- `metadata.json` — backbone HF, `max_length`, noms des classes  

Pour prédire sur de nouveaux mails : recharger le même backbone HF que dans `metadata.json`, recalculer `[CLS]` + les 3 stats sur le texte (`subject` + `body`), puis `model.predict`.

## Dépannage

- GPU saturé → `--bert-batch 8` (ou `16`)  
- CSV sans les 3 colonnes attendues → les ajouter ou adapter le script  
