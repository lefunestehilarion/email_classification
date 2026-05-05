# 📧 Email Classifier — Vue d'ensemble

Classifieur d'emails multiclasse basé sur un modèle LSTM.
Trois versions progressives, chacune autonome et utilisable indépendamment.

---

## Structure du projet

```
.
├── v0_emails_classifier.theory/     # Notebooks d'exploration et théorie
├── v1_train_predict/                # Entraînement + prédiction (base)
├── v2_eml_batch/                    # Ingestion .eml + traitement batch
└── v3_classify_csv/                 # Classification CSV sans label
```

---

## Versions

### v1 — Entraînement & Prédiction

> Point de départ. Entraîne le modèle et prédit via CLI ou API REST.

**Scripts :**
```
train.py      # Entraînement depuis zéro + extension de catégories
predict.py    # Prédiction CLI + API FastAPI
```

**Entraîner le modèle :**
```bash
python train.py train --csv data/emails.csv
# → génère models/ (lstm_model.keras, tokenizer, encoder, meta.json)
```

**Ajouter une catégorie sans réentraîner :**
```bash
python train.py extend --csv data/new_category.csv --extra-epochs 5
```

**Prédire un email :**
```bash
# CLI
python predict.py --subject "Budget Q3" --body "See attached report"

# API
uvicorn predict:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"subject": "Budget Q3", "body": "See attached report."}'
```

---

### v2 — Ingestion .eml & Batch JSON

> Ajoute la capacité de lire des fichiers `.eml`, les convertir en JSON classifiés,
> et traiter des dossiers entiers de fichiers JSON.

**Scripts ajoutés :**
```
eml_ingest.py       # Convertit .eml → JSON et classe automatiquement
batch_classify.py   # Classe un dossier de fichiers JSON existants
```

**Convertir + classer un dossier de .eml :**
```bash
python eml_ingest.py --input data/raw_eml/ --output data/inbox/
# → data/inbox/finance/mail_001.json
# → data/inbox/sport/mail_002.json   (organisés par catégorie)
```

**Surveiller un dossier en temps réel :**
```bash
python eml_ingest.py --input data/raw_eml/ --output data/inbox/ --watch
```

**Intégration Postfix (réception en temps réel) :**
```bash
cat mail.eml | python eml_ingest.py --stdin --api http://localhost:8000
```

**Classer un dossier de JSON existants :**
```bash
python batch_classify.py --input data/inbox/ --output results/classified.csv
```

---

### v3 — Classification CSV sans label

> Ajoute la classification directe d'un CSV `subject,body` sans colonne label.
> Résultat enrichi avec la catégorie prédite et le top 3 des probabilités.

**Script ajouté :**
```
classify_csv.py     # Classe un CSV et génère un CSV de sortie enrichi
```

**Classifier un CSV :**
```bash
# Local
python classify_csv.py --input data/emails.csv

# Via l'API
python classify_csv.py --input data/emails.csv --api http://localhost:8000
```

**Format d'entrée :**
```
subject,body
"Budget Q3","Please find attached..."
"Match results","Liverpool won 3-1..."
```

**Format de sortie (`emails_classified.csv`) :**
```
subject,body,prediction,confidence,top2,top2_conf,top3,top3_conf
"Budget Q3","...",finance,87.32%,work/airbus,8.14%,school,2.54%
```

---

## Docker (commun aux 3 versions)

```bash
# Build
docker build -t email-classifier .

# Entraîner
docker compose --profile training run --rm trainer

# Ajouter des catégories
docker compose --profile training run --rm extender

# Lancer l'API
docker compose up api
```

---

## Modèle

```
Embedding (3000 → 64 dims)
    └── LSTM (64, return_sequences=True)
        └── LSTM (32)
            └── Dropout (0.3)
                └── Dense (64, relu)
                    └── Dense (N classes, softmax)
```

| Paramètre | Valeur |
|-----------|--------|
| Longueur de séquence | 95e percentile du corpus |
| Loss | sparse_categorical_crossentropy |
| Métrique | Accuracy + Macro-Precision |
| Catégories | games, sport, family, work/airbus, human resources, pets, school, software engineering, finance |

---

## Choisir la bonne version

| Besoin | Version |
|--------|---------|
| Entraîner / prédire un email | v1 |
| Traiter des `.eml` ou surveiller un dossier | v2 |
| Classifier un CSV existant sans label | v3 |
| Ajouter une catégorie | v1 → `train.py extend` |
| Déployer en production | n'importe laquelle → `docker compose up api` |

---

## Étendre le projet

| Besoin | Action |
|--------|--------|
| Nouvelles catégories | `python train.py extend --csv new_data.csv` |
| Réentraîner depuis zéro | `python train.py train --csv dataset_complet.csv` |
| Emails en français | Remplacer DistilBERT par `camembert-base` dans `train.py` |
| Activer le GPU | Remplacer `tensorflow-cpu` par `tensorflow` dans `requirements.txt` |
| Changer le port API | `uvicorn predict:app --port 9000` |
