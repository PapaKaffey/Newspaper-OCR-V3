#!/usr/bin/env python3
"""
scripts/run_ocr.py — command-line interface for OCR pipeline

Usage:
    python scripts/run_ocr.py ./scans ./ocr_out --config configs/default.yml
"""
from __future__ import annotations
import argparse
from pathlib import Path
from newspaper_ocr import run_ocr


def main():
    ap = argparse.ArgumentParser(description="OCR all JP2/TIFF/PNG in INPUT_DIR")
    ap.add_argument("input_dir", help="Folder with scans")
    ap.add_argument("output_dir", help="Where to write .txt results")
    ap.add_argument("--config", help="Optional YAML path")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()
    run_ocr(Path(args.input_dir), Path(args.output_dir), args.config, args.log_dir)


if __name__ == "__main__":
    main()
