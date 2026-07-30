# Tri de mails par zero-shot (catégories dynamiques, CSV)

Trois scripts indépendants de toute base SQLite : classification, feedback,
et fine-tuning bon marché (LoRA) du modèle NLI (`facebook/bart-large-mnli`).

## Format attendu

Un CSV avec au minimum les colonnes :

```
id,subject,body
1,"Facture EDF de mars","Bonjour, veuillez trouver ci-joint..."
2,"Relance projet X","Suite à notre échange..."
```

La colonne `id` est optionnelle (l'index de ligne est utilisé sinon).

## Installation (uv)

Si `pyproject.toml` (fourni ici) n'est pas déjà fusionné avec ton projet uv
existant : soit tu le remplaces/complètes tel quel, soit tu ajoutes les
dépendances à ton `pyproject.toml` actuel avec :

```bash
uv add pandas transformers torch peft
```

Puis, dans les deux cas :

```bash
uv lock      # génère/actualise uv.lock (nécessaire pour Docker, voir plus bas)
uv sync      # crée le venv et installe les dépendances
```

Chaque commande des sections suivantes s'exécute alors avec `uv run`, par
exemple `uv run python zero_shot_csv.py ...`.

## 1. Classifier ses mails — `zero_shot_csv.py`

```bash
# Catégories passées en argument
uv run python zero_shot_csv.py --input mails.csv --categories "Spam,Business,Support,Facture"

# Catégories saisies au prompt (si --categories absent)
uv run python zero_shot_csv.py --input mails.csv

# Catégories depuis un fichier (une par ligne)
uv run python zero_shot_csv.py --input mails.csv --categories-file categories.txt

# Avec un modèle fine-tuné (voir étape 3) au lieu du modèle de base
uv run python zero_shot_csv.py --input mails.csv --model models/bart-mnli-finetuned/final
```

**Options utiles :**
| Option | Rôle |
|---|---|
| `--output` | CSV de sortie (défaut : `<input>_classified.csv`) |
| `--max-chars` | Troncature du texte avant le modèle (défaut 4000) |
| `--multi-label` | Plusieurs catégories indépendantes au lieu d'un choix exclusif |

**Sortie :** le CSV d'entrée + `predicted_label` + `predicted_score`.

## 2. Corriger une prédiction — `add_feedback_csv.py`

Les corrections vont dans un `feedback.csv` auto-suffisant (il garde
subject/body en clair, donc pas besoin de reconserver le CSV classifié
d'origine pour l'étape 3).

```bash
# Correction unique
uv run python add_feedback_csv.py --classified mails_classified.csv --email-id 12 --correct "Facture"

# Import en masse : un CSV avec colonnes id,correct
uv run python add_feedback_csv.py --classified mails_classified.csv --corrections corrections.csv

# Voir le feedback en attente (pas encore utilisé pour l'entraînement)
uv run python add_feedback_csv.py --list
```

## 3. Fine-tuner à moindre coût — `fine_tune_lora.py`

`peft` est déjà dans les dépendances du projet (voir section Installation),
pas d'étape supplémentaire :

```bash
uv run python fine_tune_lora.py --feedback-csv feedback.csv --output-dir models/bart-mnli-lora
```

Ne consomme que le feedback pas encore marqué `used_in_training=1` — chaque
run ne réentraîne donc que sur les nouvelles corrections. Pour reprendre sur
un adapter déjà entraîné plutôt que repartir de zéro :

```bash
uv run python fine_tune_lora.py --feedback-csv feedback.csv \
  --output-dir models/bart-mnli-lora-v2 \
  --base-adapter models/bart-mnli-lora/final
```

**Pourquoi LoRA plutôt qu'un fine-tune complet :** seules de petites matrices
adapter (insérées dans `q_proj`/`v_proj`) sont entraînées — moins de 1 % des
407M paramètres du modèle. Le fichier sauvegardé ne pèse que quelques Mo
(au lieu de ~1,6 Go), le training est nettement plus rapide, et le modèle de
base reste intact.

**Paramètres à connaître :**
| Option | Défaut | Rôle |
|---|---|---|
| `--lora-r` | 8 | Rang des matrices LoRA (8–16 suffit en général) |
| `--lora-alpha` | 16 | Facteur d'échelle de l'adapter |
| `--lr` | 2e-4 | LoRA tolère un learning rate plus élevé qu'un full fine-tune |
| `--epochs` | 5 | LoRA tolère plus d'époques sans trop surapprendre |
| `--n-random-negatives` | 1 | Négatifs aléatoires ajoutés par exemple de feedback |
| `--no-mark-used` | — | Ne pas consommer le feedback (pour itérer/tester) |

## 4. Partir de fichiers .eml — `eml_to_csv.py`

Si tes mails sont des fichiers `.eml` (export RFC 822) plutôt qu'un CSV déjà
prêt, convertis-les d'abord :

```bash
uv run python eml_to_csv.py --input-dir ./mes_mails --output mails.csv
```

Le script prend tous les `.eml` d'un dossier, extrait `Subject` et le corps
(`text/plain` en priorité, sinon `text/html` dépouillé de ses balises), et
écrit un `mails.csv` directement compatible avec `zero_shot_csv.py`.

## Docker

Le pipeline complet (les 4 scripts + dépendances) est conteneurisé, via `uv`
à l'intérieur de l'image (pas de `pip` dans le `Dockerfile`).

**Construire l'image :** génère d'abord `uv.lock` si ce n'est pas déjà fait
(`uv lock`, voir section Installation) — le `Dockerfile` s'appuie dessus
pour une installation reproductible :

```bash
docker compose build
```

**Lancer un script :** chaque commande devient un `docker compose run` ; le
dossier `./data` (créé à côté du `docker-compose.yml`) sert d'échange pour
les `.eml`, CSV et `feedback.csv`, et `./models` conserve les fine-tunes.

```bash
mkdir -p data/eml models
cp mes_mails/*.eml data/eml/          # tes .eml dans data/eml/

docker compose run --rm zero-shot eml_to_csv.py \
  --input-dir /data/eml --output /data/mails.csv

docker compose run --rm zero-shot zero_shot_csv.py \
  --input /data/mails.csv --categories "Spam,Business,Support,Facture"

docker compose run --rm zero-shot add_feedback_csv.py \
  --classified /data/mails_classified.csv --feedback-csv /data/feedback.csv \
  --email-id 3 --correct "Business"

docker compose run --rm zero-shot fine_tune_lora.py \
  --feedback-csv /data/feedback.csv --output-dir /app/models/bart-mnli-lora
```

Le cache des modèles Hugging Face est conservé dans un volume nommé
(`hf-cache`) pour ne pas re-télécharger `bart-large-mnli` à chaque run.

**GPU (ex. sur legion, RTX 5070 Ti) :** l'image par défaut est CPU-only. Pour
du GPU, il faut repartir d'une base CUDA (ex. `nvidia/cuda:12.4.1-runtime-
ubuntu22.04`) dans le `Dockerfile`, installer torch avec l'index CUDA
correspondant, avoir `nvidia-container-toolkit` installé sur l'hôte, et
décommenter le bloc `deploy.resources` dans `docker-compose.yml`. Dis-le-moi
si tu veux que je prépare directement cette variante GPU.

## ⚠️ Limitation actuelle

`zero_shot_csv.py --model ...` charge un modèle **complet** (via
`transformers.pipeline`), pas un adapter LoRA. Après un fine-tune LoRA, il
faut donc charger le modèle de base puis appliquer l'adapter manuellement
(`PeftModel.from_pretrained(...)`) avant de reclassifier — cette intégration
n'est pas encore automatisée dans `zero_shot_csv.py`. Dites-le si vous
voulez que je l'ajoute.

## Enchaînement typique

```bash
uv run python eml_to_csv.py --input-dir ./mes_mails --output mails.csv
uv run python zero_shot_csv.py --input mails.csv --categories "Spam,Business,Support,Facture"
# → corriger les erreurs vues dans mails_classified.csv
uv run python add_feedback_csv.py --classified mails_classified.csv --email-id 7 --correct "Business"
# … répéter pour d'autres corrections …
uv run python fine_tune_lora.py --feedback-csv feedback.csv --output-dir models/bart-mnli-lora
# → reclassifier avec le modèle amélioré (une fois l'intégration adapter faite)
```

En Docker, chaque ligne ci-dessus devient `docker compose run --rm zero-shot <script> ...`
avec des chemins sous `/data` (voir section Docker plus haut).
