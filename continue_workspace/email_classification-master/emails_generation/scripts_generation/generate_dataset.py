
import json
import requests
import time
import random
import os
import numpy as np
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
#LMSTUDIO_URL  = "http://localhost:1234/v1/chat/completions"
LMSTUDIO_URL = "http://localhost:8001/v1/chat/completions"
OUTPUT_FILE   = "emails_dataset.jsonl"
PROGRESS_FILE = "emails_progress.json"

MODELS = [
        #"meta-llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct-1m",
]

MODEL_MAX_TOKENS = {
    "mistralai/mistral-7b-instruct-v0.3": 4000,
}

BASE_CATEGORIES = [
    "human resources", "family", "school", "sport", "finance",
    "software engineering", "medical", "games", "pets", "work/airbus"
]

# Expansion automatique : 10 catégories × 2 niveaux = 20 labels
CATEGORIES = [
    f"{cat}_{suffix}"
    for cat in BASE_CATEGORIES
    for suffix in ["urgent", "not_urgent"]
]

BATCH_SIZE           = 20    # 20 emails par appel LLM
BATCHES_PER_CATEGORY = 50    # 20 × 50 × 20 labels = 20 000 emails au total
MAX_TOKENS           = 8000
MAX_RETRIES          = 4
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a dataset generator. Your job is to generate realistic, varied emails for training a text classification model.
You must respond ONLY with valid JSON. No explanation, no markdown, no preamble."""


def make_user_prompt(category: str, n: int) -> str:
    """Génère le prompt utilisateur selon la catégorie fusionnée (ex: finance_urgent)."""

    if category.endswith("_not_urgent"):
        base_cat = category[:-11]
        urgency_level = "NOT urgent (routine)"
        urgency_instructions = """\
- The email must feel routine and low-pressure:
    * Informational updates, FYI, confirmations, newsletters, casual check-ins
    * No deadlines, no time pressure, no escalation
    * Absolutely NO urgency signals: no "ASAP", no "critical", no "deadline", no "immediately"
    * Tone: calm, neutral, informational, or friendly
    * Vary intensity: some purely administrative, some warm/social, some just matter-of-fact"""

    else:  # _urgent
        base_cat = category[:-7]
        urgency_level = "URGENT"
        urgency_instructions = """\
- The email MUST feel urgent through a combination of temporal and emotional signals:
    * Temporal pressure: tight deadlines ("by end of day", "before the 3pm meeting", "overdue"),
      missed SLAs, imminent consequences if no action is taken
    * Emotional/tonal urgency: ASAP, critical, action required, escalation, strong concern,
      "this cannot wait", "I need a response today"
    * At least one temporal AND one tonal signal must appear naturally in subject or body
    * Vary urgency intensity: some mildly pressing ("could you reply today?"),
      some highly critical ("production is down, we need a fix NOW")"""

    return f"""Generate {n} realistic emails for the category: "{base_cat}" — urgency level: {urgency_level}.

Each email must:
- Feel like a real email a person would write or receive
- Topic and vocabulary must clearly relate to: {base_cat}
- Vary SIGNIFICANTLY in length:
    * some emails must be very short (2-3 sentences only)
    * some emails must be medium length (1-2 paragraphs)
    * some emails must be long (3-5 paragraphs with details)
- Vary in tone: some formal, some informal
- Vary the sender type: colleague, friend, automated system, manager, stranger, etc.
- NOT mention the category name explicitly anywhere in the subject or body
{urgency_instructions}

Respond ONLY with a valid JSON array, no preamble, no explanation:
[
  {{"subject": "...", "body": "...", "category": "{category}"}},
  ...
]
"""


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def generate_batch(category: str, n: int, model: str) -> list:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_user_prompt(category, n)}
        ],
        #"temperature": round(random.uniform(0.75, 1.0), 2),
        "temperature": round(random.uniform(0.7, 1.0), 2),
        "max_tokens": MODEL_MAX_TOKENS.get(model, MAX_TOKENS),
    }
    response = requests.post(LMSTUDIO_URL, json=payload, timeout=120)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences si le modèle les ajoute malgré les instructions
    for fence in ["```json", "```"]:
        if content.startswith(fence):
            content = content[len(fence):]
    if content.endswith("```"):
        content = content[:-3]

    return json.loads(content.strip())


def print_stats(filepath: str) -> None:
    if not os.path.exists(filepath):
        return

    lengths    = []
    by_cat     = {}

    with open(filepath) as f:
        for line in f:
            try:
                email = json.loads(line)
                wc    = len(email["body"].split())
                lengths.append(wc)
                cat = email.get("category", "unknown")
                by_cat[cat] = by_cat.get(cat, 0) + 1
            except Exception:
                pass

    if not lengths:
        return

    lengths = np.array(lengths)
    print(f"\n{'═'*60}")
    print(f"📊 Stats globales ({len(lengths)} emails)")
    print(f"   Min: {lengths.min()} | Max: {lengths.max()} | "
          f"Mean: {lengths.mean():.0f} | Std: {lengths.std():.0f} mots")

    buckets = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, 9999)]
    for lo, hi in buckets:
        count = int(((lengths >= lo) & (lengths < hi)).sum())
        pct   = count / len(lengths) * 100
        bar   = "█" * int(pct / 2)
        print(f"   {lo:>4}-{hi:<4}w : {bar:<25} {count:>5} ({pct:.1f}%)")

    # Distribution par catégorie
    print(f"\n📁 Emails par label ({len(by_cat)} labels)")
    for cat in sorted(by_cat):
        count = by_cat[cat]
        tag   = "🔴" if cat.endswith("_urgent") else "🟢"
        print(f"   {tag} {cat:<35} {count:>5}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
progress = load_progress()
total    = sum(progress.get(cat, 0) * BATCH_SIZE for cat in CATEGORIES)

target = len(CATEGORIES) * BATCHES_PER_CATEGORY * BATCH_SIZE
print(f"🚀 Génération — cible : {target} emails ({len(CATEGORIES)} labels × "
      f"{BATCHES_PER_CATEGORY} batches × {BATCH_SIZE})")
print(f"   Output   : {OUTPUT_FILE}")
print(f"   Reprise  : {progress if progress else 'départ frais'}")
print(f"   Heure    : {datetime.now().strftime('%H:%M:%S')}\n")

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for category in CATEGORIES:
        batches_done = progress.get(category, 0)

        if batches_done >= BATCHES_PER_CATEGORY:
            print(f"⏭  {category} — déjà complet, on passe")
            continue

        tag = "🔴 URGENT" if category.endswith("_urgent") else "🟢 NOT URGENT"
        print(f"\n{'═'*60}")
        print(f"📁 {category.upper()}  [{tag}]  (reprise batch {batches_done + 1})")
        print(f"{'═'*60}")

        cat_count = batches_done * BATCH_SIZE

        for batch in range(batches_done, BATCHES_PER_CATEGORY):
            model   = MODELS[batch % len(MODELS)]
            success = False

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    emails = generate_batch(category, BATCH_SIZE, model)

                    valid = [e for e in emails if "subject" in e and "body" in e]
                    if not valid:
                        raise ValueError("Aucun email valide dans la réponse")

                    for email in valid:
                        email["category"] = category   # label enforced côté script
                        email["model"]    = model
                        f.write(json.dumps(email, ensure_ascii=False) + "\n")
                    f.flush()

                    cat_count += len(valid)
                    total     += len(valid)
                    progress[category] = batch + 1
                    save_progress(progress)

                    print(f"  ✅ [{model}] batch {batch + 1:>2}/{BATCHES_PER_CATEGORY}"
                          f" — +{len(valid)} emails"
                          f" | cat: {cat_count} | total: {total}")
                    success = True
                    break

                except Exception as e:
                    wait = 3 * attempt
                    print(f"  ⚠️  tentative {attempt}/{MAX_RETRIES} échouée : {e}"
                          f" — retry dans {wait}s")
                    time.sleep(wait)

            if not success:
                print(f"  ❌ Batch {batch + 1} abandonné après {MAX_RETRIES} tentatives")

            time.sleep(0.4)

        print(f"\n✔  {category} terminé — {cat_count} emails générés")

print(f"\n{'═'*60}")
print(f"🎉 Génération complète !")
print(f"   Total emails : {total}")
print(f"   Fichier      : {OUTPUT_FILE}")
print(f"   Heure        : {datetime.now().strftime('%H:%M:%S')}")
print_stats(OUTPUT_FILE)

