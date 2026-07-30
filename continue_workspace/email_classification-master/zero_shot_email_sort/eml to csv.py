#!/usr/bin/env python3
"""
eml_to_csv.py — Convertit un dossier de fichiers .eml (RFC 822) en CSV
compatible avec zero_shot_csv.py (colonnes id, subject, body).

Usage :
    python eml_to_csv.py --input-dir ./mes_mails --output mails.csv
"""
from __future__ import annotations

import argparse
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pandas as pd


def strip_html(html: str) -> str:
    """Dépouillage grossier du HTML : suffisant pour donner du texte au
    zero-shot, pas destiné à produire un rendu propre."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_body(msg) -> str:
    """Préfère la partie text/plain ; à défaut, dépouille le text/html."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                try:
                    return part.get_content().strip()
                except Exception:
                    continue
        for part in msg.walk():
            if part.get_content_type() == "text/html" and not part.get_filename():
                try:
                    return strip_html(part.get_content())
                except Exception:
                    continue
        return ""
    else:
        try:
            if msg.get_content_type() == "text/html":
                return strip_html(msg.get_content())
            return msg.get_content().strip()
        except Exception:
            return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertit un dossier de .eml en CSV (id, subject, body)")
    parser.add_argument("--input-dir", type=Path, required=True, help="Dossier contenant les .eml")
    parser.add_argument("--output", type=Path, default=Path("mails.csv"))
    parser.add_argument("--max-chars", type=int, default=20000, help="Troncature du corps avant écriture CSV")
    args = parser.parse_args()

    eml_files = sorted(args.input_dir.glob("*.eml"))
    if not eml_files:
        raise SystemExit(f"❌ Aucun .eml trouvé dans {args.input_dir}")

    rows = []
    for i, path in enumerate(eml_files):
        with open(path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        subject = str(msg.get("Subject", "")).strip()
        body = extract_body(msg)
        if len(body) > args.max_chars:
            body = body[: args.max_chars - 1] + "…"

        rows.append({"id": i, "filename": path.name, "subject": subject, "body": body})
        print(f"  [{i}] {path.name} — sujet: {subject[:60]!r}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\n✅ {len(df)} mail(s) converti(s) → {args.output}")


if __name__ == "__main__":
    main()
