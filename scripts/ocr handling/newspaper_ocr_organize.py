#!/usr/bin/env python3
"""
Organize OCR Output Files into per-page or date-based subfolders

Usage:
    python scripts/organize_ocr_files.py --input-dir "Processed OCR Text" --mode page --output-dir organized
Modes:
    page - one folder per OCR'd page
    date - folders by extracted publication date (YYYY/MM-DD)
"""

import re
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

DATE_REGEX = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}",
    re.IGNORECASE,
)


def extract_date(text):
    match = DATE_REGEX.search(text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(), "%B %d, %Y")
    except Exception:
        return None


def organize(input_dir: Path, output_dir: Path, mode: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    txt_files = list(input_dir.glob("*.txt"))

    for txt_file in txt_files:
        page_id = txt_file.stem
        meta_path = input_dir / f"{page_id}_meta.json"

        if not meta_path.exists():
            print(f"⚠️ Skipping {page_id} — no meta file found.")
            continue

        text = txt_file.read_text(encoding='utf-8', errors='ignore')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))

        date_obj = extract_date(text)
        folder = None

        if mode == "date" and date_obj:
            folder = output_dir / f"{date_obj.year}" / date_obj.strftime("%m-%d")
        else:
            # fallback to per-page folder
            folder = output_dir / page_id

        folder.mkdir(parents=True, exist_ok=True)

        # Move files
        shutil.copy(txt_file, folder / txt_file.name)
        shutil.copy(meta_path, folder / meta_path.name)

        # Optionally include enriched output if exists
        enriched_path = input_dir / f"{page_id}_enriched.json"
        if enriched_path.exists():
            shutil.copy(enriched_path, folder / enriched_path.name)

        manifest[page_id] = {
            "original_path": str(txt_file),
            "new_path": str(folder),
            "date": date_obj.strftime("%Y-%m-%d") if date_obj else None
        }

        print(f"✅ Moved {page_id} → {folder}")

    # Save manifest
    with open(output_dir / "manifest.json", "w", encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📜 Created manifest with {len(manifest)} entries at {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Path to flat OCR output")
    parser.add_argument("--output-dir", required=True, help="Where to organize files")
    parser.add_argument("--mode", choices=["page", "date"], default="page", help="Organize by page or date")
    args = parser.parse_args()

    organize(Path(args.input_dir), Path(args.output_dir), args.mode)
