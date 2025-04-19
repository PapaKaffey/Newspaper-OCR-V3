#!/usr/bin/env python3
"""
scripts/download_jp2.py — simple CLI for batch JP2/TIFF downloading

Usage:
    python scripts/download_jp2.py ./downloads https://nebnewspapers.unl.edu/data/batches/batch_nbu_plattsmouth01_ver01/
"""
from __future__ import annotations
import argparse
from pathlib import Path
from newspaper_ocr import JP2Downloader


def main():
    ap = argparse.ArgumentParser(description="Download JP2/TIFF batches")
    ap.add_argument("output_dir", help="Destination folder")
    ap.add_argument("urls", nargs="+", help="Batch URLs")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    JP2Downloader(Path(args.output_dir), max_workers=args.workers).download_all_batches(args.urls)


if __name__ == "__main__":
    main()
