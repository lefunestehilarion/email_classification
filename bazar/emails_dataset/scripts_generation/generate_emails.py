import json
import requests
import time
import random
import os
import numpy as np
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
LMSTUDIO_URL  = "http://localhost:1234/v1/chat/completions"
OUTPUT_FILE   = "emails_dataset.jsonl"
PROGRESS_FILE = "emails_progress.json"  # pour reprendre si crash

MODELS = [
    "meta-llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct-1m",
]

# Mistral has a smaller context window — cap its tokens to avoid 400 errors
MODEL_MAX_TOKENS = {
    "mistralai/mistral-7b-instruct-v0.3": 4000,
}

CATEGORIES = [
    "human resources", "family", "school", "sport", "finance",
    "software engineering", "medical", "games", "pets", "work/airbus"
]

BATCH_SIZE           = 20   # 20 emails par appel → plus de tokens par email
BATCHES_PER_CATEGORY = 50   # 20 × 50 = 1000 par catégorie → 10 000 au total
MAX_TOKENS           = 8000 # large pour permettre les longs emails
MAX_RETRIES          = 4    # tentatives avant de passer au batch suivant
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a dataset generator. Your job is to generate realistic, varied emails for training a text classification model.
You must respond ONLY with valid JSON. No explanation, no markdown, no preamble."""

def make_user_prompt(category, n):
    return f"""Generate {n} realistic emails for the category: "{category}".

Each email must:
- Feel like a real email a person would write or receive
- Vary SIGNIFICANTLY in length:
    * some emails must be very short (2-3 sentences only)
    * some emails must be medium length (1-2 paragraphs)
    * some emails must be long (3-5 paragraphs with details)
- Vary in tone: some formal, some informal, some urgent, some casual
- Vary the sender type: colleague, friend, automated system, manager, stranger, etc.
- Contain vocabulary and context clearly related to the category
- NOT mention the category name explicitly anywhere in the subject or body

Respond ONLY with a valid JSON array, no preamble, no explanation:
[
  {{"subject": "...", "body": "...", "category": "{category}"}},
  ...
]
"""

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

def generate_batch(category, n, model):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_user_prompt(category, n)}
        ],
        "temperature": round(random.uniform(0.75, 1.0), 2),
        "max_tokens": MODEL_MAX_TOKENS.get(model, MAX_TOKENS),
    }
    response = requests.post(LMSTUDIO_URL, json=payload, timeout=120)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    # Strip markdown fences if model ignores instructions
    content = content.strip()
    for fence in ["```json", "```"]:
        if content.startswith(fence):
            content = content[len(fence):]
    if content.endswith("```"):
        content = content[:-3]
    return json.loads(content.strip())

def print_stats(filepath):
    if not os.path.exists(filepath):
        return
    lengths = []
    with open(filepath) as f:
        for line in f:
            try:
                email = json.loads(line)
                lengths.append(len(email["body"].split()))
            except:
                pass
    if not lengths:
        return
    lengths = np.array(lengths)
    print(f"\n📊 Length stats so far ({len(lengths)} emails):")
    print(f"   Min: {lengths.min()} | Max: {lengths.max()} | Mean: {lengths.mean():.0f} | Std: {lengths.std():.0f} words")
    buckets = [(0,20),(20,50),(50,100),(100,200),(200,500),(500,9999)]
    for lo, hi in buckets:
        count = int(((lengths >= lo) & (lengths < hi)).sum())
        pct   = count / len(lengths) * 100
        bar   = "█" * int(pct / 2)
        print(f"   {lo:>4}-{hi:<4}w : {bar:<25} {count:>5} ({pct:.1f}%)")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
progress = load_progress()  # {category: batches_done}
total    = sum(progress.get(cat, 0) * BATCH_SIZE for cat in CATEGORIES)

print(f"🚀 Starting generation — target: {len(CATEGORIES) * BATCHES_PER_CATEGORY * BATCH_SIZE} emails")
print(f"   Output : {OUTPUT_FILE}")
print(f"   Resume : {progress if progress else 'fresh start'}")
print(f"   Time   : {datetime.now().strftime('%H:%M:%S')}\n")

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for category in CATEGORIES:
        batches_done = progress.get(category, 0)

        if batches_done >= BATCHES_PER_CATEGORY:
            print(f"⏭  {category} already complete — skipping")
            continue

        print(f"\n{'═'*60}")
        print(f"📁 Category: {category.upper()}  (resuming at batch {batches_done+1})")
        print(f"{'═'*60}")

        cat_count = batches_done * BATCH_SIZE

        for batch in range(batches_done, BATCHES_PER_CATEGORY):
            model   = MODELS[batch % len(MODELS)]
            success = False

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    emails = generate_batch(category, BATCH_SIZE, model)

                    # Validate every email has required fields
                    valid = [e for e in emails if "subject" in e and "body" in e]
                    if not valid:
                        raise ValueError("No valid emails in response")

                    for email in valid:
                        email["category"] = category  # enforce correct label
                        email["model"]    = model
                        f.write(json.dumps(email, ensure_ascii=False) + "\n")
                    f.flush()

                    cat_count += len(valid)
                    total     += len(valid)
                    progress[category] = batch + 1
                    save_progress(progress)

                    print(f"  ✅ [{model}] batch {batch+1:>2}/{BATCHES_PER_CATEGORY}"
                          f" — +{len(valid)} emails"
                          f" | cat: {cat_count} | total: {total}")
                    success = True
                    break

                except Exception as e:
                    wait = 3 * attempt
                    print(f"  ⚠️  attempt {attempt}/{MAX_RETRIES} failed: {e} — retrying in {wait}s")
                    time.sleep(wait)

            if not success:
                print(f"  ❌ Batch {batch+1} skipped after {MAX_RETRIES} attempts")

            time.sleep(0.4)

        print(f"\n✔  {category} done — {cat_count} emails generated")

print(f"\n{'═'*60}")
print(f"🎉 Generation complete!")
print(f"   Total emails : {total}")
print(f"   Output file  : {OUTPUT_FILE}")
print(f"   Time         : {datetime.now().strftime('%H:%M:%S')}")
print_stats(OUTPUT_FILE)
