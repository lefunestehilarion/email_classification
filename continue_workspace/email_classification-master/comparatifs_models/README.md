# Amélioration du zero-shot par feedback utilisateur + benchmark de rapidité

Ce dossier complète le pipeline existant (LSTM supervisé + zero-shot NLI) avec :

1. une table pour stocker les corrections utilisateur sur les prédictions zero-shot,
2. un script pour fine-tuner le modèle NLI sur ces corrections,
3. un script de benchmark comparant la rapidité du LSTM et du zero-shot.

## Contexte : deux approches très différentes

| | LSTM (`train.py`, `predict.py`, …) | Zero-shot (`test_mdeberta_zero_shot.py`) |
|---|---|---|
| Entraînement | Supervisé sur un CSV labellisé | Aucun — modèle pré-entraîné générique |
| Mécanisme | Une passe → softmax sur N catégories | Une hypothèse "This email is about {label}." testée **par label candidat** |
| Nouvelle catégorie | Nécessite un réentraînement | Fonctionne sans rien faire (mais moins précis) |
| Vitesse attendue | Rapide | Plus lent (N appels modèle par email) |

## Comment marche le zero-shot, concrètement

`bart-large-mnli` n'est pas un classifieur d'emails : c'est un modèle de **NLI** (Natural Language Inference), entraîné à répondre à la question *"la phrase B découle-t-elle logiquement de la phrase A ?"* avec trois réponses possibles : `entailment`, `neutral`, `contradiction`.

Le script `test_mdeberta_zero_shot.py` exploite ça en construisant, pour chaque label candidat, une hypothèse du type :

```
This email is about {label}.
```

Le modèle teste cette hypothèse contre le texte de l'email. Le label dont l'hypothèse obtient le score d'`entailment` le plus haut devient la prédiction. Le modèle généralise depuis ce qu'il a appris sur des millions de paires de phrases — il n'a jamais vu un seul de tes emails.

## 1. La table `feedback`

Ajoutée dans `models.py`, aux côtés de `Email` et `Label` existantes :

```python
class Feedback(Base):
    __tablename__ = "feedback"

    id: int                  # clé primaire
    email_id: int            # FK vers emails.id
    label_predit: str        # ce que le modèle avait prédit
    label_correct: str       # la vraie catégorie (correction utilisateur)
    score_predit: float|None # score de confiance de la prédiction d'origine
    model_id: str|None       # quel modèle a fait la prédiction (utile si tu compares plusieurs versions)
    created_at: datetime
    used_in_training: int    # 0/1 — évite de réutiliser un feedback déjà appris
```

`label_predit`/`label_correct` sont stockés en texte libre plutôt qu'en clé étrangère stricte, pour rester robustes si un label est renommé.

## 2. Enregistrer du feedback — `add_feedback.py`

**Correction ponctuelle :**
```bash
python add_feedback.py --email-id 42 --predicted "Spam" --correct "Business" --score 0.55
```

**Import en masse depuis un CSV** (colonnes : `email_id,label_predit,label_correct[,score_predit][,model_id]`) :
```bash
python add_feedback.py --csv corrections.csv
```

**Voir le feedback en attente** (pas encore utilisé pour un entraînement) :
```bash
python add_feedback.py --list
```

Toutes les commandes acceptent `--db chemin/vers/emails.db` (défaut : `data/emails.db`).

## 3. Fine-tuner le zero-shot — `fine_tune_zero_shot.py`

### Principe

Plutôt que de remplacer le mécanisme NLI par une tête de classification classique (ce qui ferait perdre la capacité zero-shot sur une catégorie jamais vue), le script continue d'entraîner le modèle **sur la tâche NLI elle-même**, avec des exemples construits depuis le feedback :

Pour chaque ligne de feedback non utilisée :
- **1 exemple positif** : `(email, "This email is about {label_correct}.", entailment)`
- **1 exemple négatif "dur"** : `(email, "This email is about {label_predit}.", contradiction)` — l'erreur précise que le modèle a faite
- **N exemples négatifs aléatoires** : d'autres labels au hasard, pour que le modèle ne fasse pas que mémoriser cette seule confusion et continue de bien discriminer le reste

Le modèle reste un classifieur NLI généraliste — juste mieux calibré sur ton vocabulaire métier et tes ambiguïtés fréquentes — tout en gardant l'aptitude zero-shot pour des catégories que tu n'as jamais corrigées.

### Usage

```bash
# Fine-tuner sur tout le feedback non encore utilisé
python fine_tune_zero_shot.py --db data/emails.db --output-dir models/bart-mnli-finetuned

# Plus de négatifs aléatoires par exemple (meilleure couverture de l'espace des labels)
python fine_tune_zero_shot.py --db data/emails.db --output-dir models/bart-mnli-finetuned --n-random-negatives 2

# Reprendre depuis un fine-tune précédent plutôt que depuis le modèle de base
python fine_tune_zero_shot.py --db data/emails.db --output-dir models/bart-mnli-finetuned-v2 --base-model models/bart-mnli-finetuned/final
```

### Options principales

| Option | Défaut | Rôle |
|---|---|---|
| `--base-model` | `facebook/bart-large-mnli` | Modèle de départ (HF id ou chemin local) |
| `--epochs` | 3 | Nombre d'époques |
| `--batch-size` | 4 | Taille de batch (à ajuster selon ta VRAM) |
| `--lr` | 1e-5 | Taux d'apprentissage |
| `--n-random-negatives` | 1 | Nb de labels aléatoires utilisés comme négatifs en plus du négatif "dur" |
| `--val-ratio` | 0.15 | Part du feedback réservée à la validation |
| `--no-mark-used` | — | Ne marque pas le feedback comme consommé (utile pour itérer/tester sans le "brûler") |

### Résultat

Le modèle fine-tuné est sauvegardé dans `{output-dir}/final/`. Utilise ce chemin comme `MODEL_ID` dans `test_mdeberta_zero_shot.py` (ou comme `--zero-shot-model` dans le benchmark ci-dessous) pour l'utiliser à la place du modèle de base.

À la fin du run, le feedback utilisé est marqué `used_in_training=1` dans la base, pour ne pas le réapprendre indéfiniment au prochain fine-tuning.

### Ce qui manque pour tourner en pratique

Ce script suppose `transformers` et `torch` installés (`pip install transformers torch`), et tourne idéalement avec un GPU — sur CPU le fine-tuning de `bart-large-mnli` (modèle ~400M paramètres) sera lent même sur un petit volume de feedback.

## 4. Comparer la rapidité — `benchmark_speed.py`

Mesure, sur le **même jeu d'emails**, pour le LSTM et le zero-shot :

- **Latence par email** : moyenne, médiane, p95 (en millisecondes) — mode "un email à la fois", représentatif d'un usage en temps réel.
- **Débit** : emails traités par seconde. Pour le LSTM, mesuré aussi en vrai mode batch (toutes les requêtes envoyées au modèle en une fois, comme le fait `classify_csv.py`). Pour le zero-shot, le pipeline HF ne batch pas plusieurs emails plus efficacement qu'un par un sur CPU — il batch surtout les labels candidats en interne — donc le débit reporté correspond au mode séquentiel, mesuré explicitement plutôt que supposé.

### Usage

```bash
# Benchmark standard sur 50 emails pris dans la base
python benchmark_speed.py --db data/emails.db --model-dir models --n-emails 50

# Comparer le zero-shot fine-tuné au modèle de base
python benchmark_speed.py --db data/emails.db --model-dir models --zero-shot-model models/bart-mnli-finetuned/final

# Sauvegarder les résultats en CSV
python benchmark_speed.py --db data/emails.db --model-dir models --output results/benchmark.csv

# Ne tester qu'un seul des deux modèles
python benchmark_speed.py --db data/emails.db --skip-lstm        # zero-shot seul
python benchmark_speed.py --db data/emails.db --skip-zero-shot   # LSTM seul
```

### Sortie

Un tableau récapitulatif :

```
Modèle                          Setup(s)  Lat.moy(ms)  Lat.médiane(ms)  Lat.p95(ms)  Débit(email/s)
-------------------------------  --------  -----------  ---------------  -----------  --------------
LSTM (supervisé)                 2.31      4.2          3.9              6.1          238.1
Zero-shot NLI (facebook/...)     8.04      312.5        298.2            410.3        3.2

👉 Le LSTM est environ 74.4x plus rapide que le zero-shot sur ce run.
```

L'écart attendu vient du mécanisme : le zero-shot évalue une hypothèse par label candidat à chaque email, alors que le LSTM fait une seule passe avant softmax sur toutes les catégories en même temps.

## Workflow complet recommandé

```bash
# 1. Utiliser le zero-shot en production, repérer les erreurs
python test_mdeberta_zero_shot.py --db data/emails.db --details 20

# 2. Enregistrer les corrections au fil de l'eau
python add_feedback.py --email-id 42 --predicted "Spam" --correct "Business" --score 0.55
# ... ou en masse via CSV

# 3. Une fois assez de feedback accumulé, fine-tuner
python add_feedback.py --list   # vérifier le volume disponible
python fine_tune_zero_shot.py --db data/emails.db --output-dir models/bart-mnli-finetuned

# 4. Comparer rapidité + pertinence avant/après
python benchmark_speed.py --db data/emails.db --model-dir models \
    --zero-shot-model models/bart-mnli-finetuned/final

# 5. Mettre à jour MODEL_ID dans test_mdeberta_zero_shot.py vers le modèle fine-tuné
```

## Statut des tests effectués

Validé dans l'environnement de développement :
- création des tables (`models.py`),
- enregistrement de feedback ponctuel et en masse (`add_feedback.py`),
- logique de génération des exemples NLI à partir du feedback (positifs/négatifs durs/négatifs aléatoires).


