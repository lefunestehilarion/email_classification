import json
import csv
import os

INPUT_FILE  = "emails_dataset.jsonl"
OUTPUT_FILE = "emails_dataset.csv"

count = 0
with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as fout:

    writer = csv.writer(fout)
    writer.writerow(["subject", "body", "label"])  # header — same as original CSV

    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            email = json.loads(line)
            subject  = email.get("subject", "").strip()
            body     = email.get("body", "").strip()
            label    = email.get("category", "").strip()
            if subject and body and label:
                writer.writerow([subject, body, label])
                count += 1
        except json.JSONDecodeError as e:
            print(f"⚠️  Skipped malformed line: {e}")

print(f"✅ Done — {count} emails written to {OUTPUT_FILE}")
