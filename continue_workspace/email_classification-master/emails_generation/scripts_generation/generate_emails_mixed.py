import json
import requests
import time
import random
import os
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
LMSTUDIO_URL  = "http://localhost:1234/v1/chat/completions"
OUTPUT_FILE   = "emails_dataset.jsonl"   # appended to existing dataset
PROGRESS_FILE = "emails_progress_mixed.json"

MODELS = [
    "meta-llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct-1m",
]

MODEL_MAX_TOKENS = {
    "mistralai/mistral-7b-instruct-v0.3": 4000,
}

# Each entry: (dominant_label, secondary_context, description for the prompt)
MIXED_PAIRS = [
    ("sport",            "medical",           "sports injury, recovery, physical therapy, or performance health"),
    ("medical",          "work/airbus",        "occupational health, workplace accident, or industrial injury at an aerospace company"),
    ("human resources",  "medical",            "employee sick leave, medical certificates, or health insurance reimbursement at work"),
    ("family",           "finance",            "household budget, family expenses, inheritance, or money between family members"),
    ("family",           "school",             "parent contacting school, child's academic performance, or school fees"),
    ("work/airbus",      "software engineering","software bug, IT issue, or technical tool used in an aerospace engineering context"),
    ("games",            "work/airbus",         "company gaming event, team building through games, or gamification at work"),
    ("finance",          "software engineering","fintech, billing system bug, software subscription cost, or IT budget"),
    ("pets",             "medical",            "veterinary care, pet medication, or animal health concern"),
    ("school",           "sport",              "school sports team, physical education, or student athlete"),
    ("human resources",  "family",             "parental leave, family emergency at work, or childcare policy"),
    ("games",            "family",             "family game night, children's video games, or gaming as a family activity"),
]

BATCH_SIZE           = 15   # smaller batches — mixed prompts are more complex
BATCHES_PER_PAIR     = 10   # 15 × 10 = 150 per pair × 12 pairs = 1800 emails
MAX_TOKENS           = 8000
MAX_RETRIES          = 4
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a dataset generator creating realistic emails for training a text classification model.
You must respond ONLY with valid JSON. No explanation, no markdown, no preamble."""

def make_mixed_prompt(dominant, secondary, description, n):
    return f"""Generate {n} realistic emails that are primarily about "{dominant}" but also naturally involve "{secondary}" context.

The emails should feel like real emails where the topic naturally blends both domains: {description}.

Each email must:
- Be CLEARLY dominated by the "{dominant}" topic — that is the correct label
- Naturally incorporate vocabulary and context from "{secondary}" without forcing it
- Vary SIGNIFICANTLY in length:
    * some very short (2-3 sentences)
    * some medium (1-2 paragraphs)
    * some long (3-5 paragraphs)
- Vary in tone: formal, informal, urgent, casual
- Vary sender type: colleague, friend, automated system, manager, patient, coach, etc.
- NOT explicitly name either category

Respond ONLY with a valid JSON array:
[
  {{"subject": "...", "body": "...", "category": "{dominant}", "mixed_with": "{secondary}"}},
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

def generate_batch(dominant, secondary, description, n, model):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_mixed_prompt(dominant, secondary, description, n)}
        ],
        "temperature": round(random.uniform(0.75, 1.0), 2),
        "max_tokens":  MODEL_MAX_TOKENS.get(model, MAX_TOKENS),
    }
    response = requests.post(LMSTUDIO_URL, json=payload, timeout=120)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    content = content.strip()
    for fence in ["```json", "```"]:
        if content.startswith(fence):
            content = content[len(fence):]
    if content.endswith("```"):
        content = content[:-3]
    return json.loads(content.strip())

# ─── MAIN ─────────────────────────────────────────────────────────────────────
progress = load_progress()
total    = 0

print(f"🚀 Generating mixed-category emails")
print(f"   Target : ~{len(MIXED_PAIRS) * BATCHES_PER_PAIR * BATCH_SIZE} emails")
print(f"   Output : {OUTPUT_FILE}  (appended)")
print(f"   Time   : {datetime.now().strftime('%H:%M:%S')}\n")

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for (dominant, secondary, description) in MIXED_PAIRS:
        pair_key     = f"{dominant}+{secondary}"
        batches_done = progress.get(pair_key, 0)

        if batches_done >= BATCHES_PER_PAIR:
            print(f"⏭  {pair_key} already complete — skipping")
            continue

        print(f"\n{'═'*60}")
        print(f"📁 {dominant.upper()}  ←  {secondary}  (batch {batches_done+1}/{BATCHES_PER_PAIR})")
        print(f"{'═'*60}")

        pair_count = batches_done * BATCH_SIZE

        for batch in range(batches_done, BATCHES_PER_PAIR):
            model   = MODELS[batch % len(MODELS)]
            success = False

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    emails = generate_batch(dominant, secondary, description, BATCH_SIZE, model)

                    valid = [e for e in emails if "subject" in e and "body" in e]
                    if not valid:
                        raise ValueError("No valid emails in response")

                    for email in valid:
                        email["category"]   = dominant   # enforce dominant label
                        email["mixed_with"] = secondary  # keep trace of secondary
                        email["model"]      = model
                        f.write(json.dumps(email, ensure_ascii=False) + "\n")
                    f.flush()

                    pair_count += len(valid)
                    total      += len(valid)
                    progress[pair_key] = batch + 1
                    save_progress(progress)

                    print(f"  ✅ [{model}] batch {batch+1:>2}/{BATCHES_PER_PAIR}"
                          f" — +{len(valid)} | pair: {pair_count} | total: {total}")
                    success = True
                    break

                except Exception as e:
                    wait = 3 * attempt
                    print(f"  ⚠️  attempt {attempt}/{MAX_RETRIES} failed: {e} — retrying in {wait}s")
                    time.sleep(wait)

            if not success:
                print(f"  ❌ Batch {batch+1} skipped after {MAX_RETRIES} attempts")

            time.sleep(0.4)

        print(f"\n✔  {pair_key} done — {pair_count} emails")

print(f"\n{'═'*60}")
print(f"🎉 Done! {total} mixed emails added to {OUTPUT_FILE}")
print(f"   Time : {datetime.now().strftime('%H:%M:%S')}")
