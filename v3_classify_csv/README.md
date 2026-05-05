# 📧 Email Classifier

Classifieur d'emails multiclasse basé sur un modèle LSTM.  
Extensible à de nouvelles catégories sans réentraîner depuis zéro.

---

## Structure du projet

```
email_classifier/
├── train.py          # Entraînement + extension de catégories
├── predict.py        # Prédiction CLI + API REST (FastAPI)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── emails.csv            # Dataset principal (subject, body, label)
│   └── new_categories.csv    # Nouvelles catégories à ajouter
└── models/                   # Artefacts générés après entraînement
    ├── lstm_model.keras
    ├── keras_tokenizer.pkl
    ├── tfidf_vectorizer.pkl
    ├── label_encoder.pkl
    └── meta.json
```

---

## Format du CSV

Le fichier CSV doit contenir exactement ces 3 colonnes :

```
subject,body,label
Budget Q3,Please find attached the quarterly report,finance
Match results,Liverpool won 3-1 against Arsenal,sport
...
```

Catégories actuelles : `games`, `sport`, `family`, `work/airbus`,  
`human resources`, `pets`, `school`, `software engineering`, `finance`

---

## Utilisation locale

### 1. Installer les dépendances

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 2. Entraîner le modèle

```bash
python train.py train --csv data/emails.csv
```

Options disponibles :

| Option | Défaut | Description |
|--------|--------|-------------|
| `--csv` | *(requis)* | Chemin vers le CSV |
| `--output-dir` | `models` | Dossier de sortie |
| `--epochs` | `15` | Nombre d'époques |
| `--max-features` | `3000` | Taille du vocabulaire TF-IDF / Keras |
| `--test-size` | `0.2` | Part des données de test |

### 3. Prédire en CLI

```bash
# Avec subject + body séparés
python predict.py --subject "Budget Q3" --body "See attached report"

# Avec texte brut
python predict.py --text "Reunion equipe demain 9h"

# Top 5 au lieu de 3
python predict.py --text "New game release" --top-k 5
```

Sortie :
```
📧 Prédiction  : finance
🎯 Confiance   : 87.32%

📊 Top 3 :
   finance                   ██████████████████████████ 87.32%
   work/airbus               ███ 8.14%
   school                    █ 2.54%
```

### 4. Lancer l'API REST

```bash
uvicorn predict:app --host 0.0.0.0 --port 8000
```

**Endpoints :**

```
GET  /health    → statut + liste des catégories
POST /predict   → prédiction
```

**Exemple de requête :**

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"subject": "Budget Q3", "body": "See attached report.", "top_k": 3}'
```

**Réponse :**

```json
{
  "prediction": "finance",
  "confidence": 0.8732,
  "top_k": [
    {"label": "finance",     "confidence": 0.8732},
    {"label": "work/airbus", "confidence": 0.0814},
    {"label": "school",      "confidence": 0.0254}
  ]
}
```

Documentation interactive : [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Ajouter de nouvelles catégories

Sans réentraîner depuis zéro — seulement un fine-tuning de la tête du modèle.

### 1. Préparer un CSV avec UNIQUEMENT les nouveaux emails

```
subject,body,label
My cat is sick,She stopped eating since yesterday,pets_health
...
```

### 2. Lancer l'extension

```bash
python train.py extend \
    --csv data/new_categories.csv \
    --extra-epochs 5
```

Le modèle existant dans `models/` est mis à jour automatiquement.  
Les anciennes catégories sont conservées.

---

## Utilisation avec Docker

### Build de l'image

```bash
docker build -t email-classifier .
```

### Entraîner via Docker

```bash
docker run --rm \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/models:/app/models \
    email-classifier \
    python train.py train --csv /app/data/emails.csv
```

### Lancer l'API via Docker

```bash
docker run -d \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models:ro \
    --name email_api \
    email-classifier
```

### Via Docker Compose

```bash
# Entraîner
docker compose --profile training run --rm trainer

# Ajouter des catégories
docker compose --profile training run --rm extender

# Lancer l'API
docker compose up api
```

---

## Architecture du modèle

```
Input (séquence de tokens)
    │
Embedding (3000 → 64 dims)
    │
LSTM (64 unités, return_sequences=True)
    │
LSTM (32 unités)
    │
Dropout (0.3)
    │
Dense (64, relu)
    │
Dense (N classes, softmax)
```

- **Longueur de séquence** : calculée dynamiquement au 95e percentile du corpus
- **Loss** : `sparse_categorical_crossentropy`
- **Métriques** : Accuracy + Macro-Precision (équitable entre classes)

---

## Étendre le projet

| Besoin | Action |
|--------|--------|
| Nouvelles catégories | `python train.py extend --csv new_data.csv` |
| Dataset plus grand | `python train.py train --csv big_dataset.csv --epochs 20` |
| Emails en français | Remplacer DistilBERT par `camembert-base` dans train.py |
| GPU | Remplacer `tensorflow-cpu` par `tensorflow` dans requirements.txt |
| Changer le port API | `uvicorn predict:app --port 9000` ou modifier `docker-compose.yml` |
