#!/usr/bin/env python3
"""
newspaper_ocr.jp2_downloader
============================
Download and mirror JP2/TIFF/PNG files from OpenONI-style HTTP directories.

Supports recursive traversal of public newspaper archives such as:
https://nebnewspapers.unl.edu/data/batches/

Example
-------
>>> from newspaper_ocr import JP2Downloader
>>> urls = [
...     "https://nebnewspapers.unl.edu/data/batches/batch_nbu_plattsmouth01_ver01/",
... ]
>>> JP2Downloader("downloads", max_workers=4).download_all_batches(urls)
"""

from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence, List, Dict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

__all__ = ["JP2Downloader"]


class JP2Downloader:
    """Recursively download JP2, TIFF, or PNG files from HTTP directory listings."""

    SUPPORTED_EXTS = (".jp2", ".tif", ".tiff", ".png")

    def __init__(
        self,
        base_dir: str | Path,
        *,
        max_workers: int = 4,
        chunk_size: int = 8192,
        retries: int = 3,
    ) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.failed: List[Dict[str, str]] = []

        # Session with retry logic
        self.session = requests.Session()
        retry_cfg = Retry(
            total=retries,
            backoff_factor=0.8,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry_cfg))
        self.session.mount("http://", HTTPAdapter(max_retries=retry_cfg))

        # Logging setup
        self.logger = logging.getLogger("jp2_downloader")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

    def download_all_batches(self, batch_urls: Sequence[str]) -> None:
        """Download all image files across one or more batch root URLs."""
        for idx, url in enumerate(batch_urls, 1):
            self.logger.info("(%d/%d) Scanning: %s", idx, len(batch_urls), url)
            self._download_batch(url.rstrip("/"))

        if self.failed:
            fail_path = self.base_dir / "failed_downloads.txt"
            with fail_path.open("w", encoding="utf-8") as fh:
                for item in self.failed:
                    fh.write(f"{item['url']}\t{item['error']}\n")
            self.logger.warning(
                "Downloaded with %d failures – see: %s", len(self.failed), fail_path
            )
        else:
            self.logger.info("✅ All downloads completed successfully!")

    def _download_batch(self, batch_url: str) -> None:
        batch_name = Path(batch_url).name
        local_root = self.base_dir / batch_name

        jp2_urls = self._find_jp2_files(batch_url)
        if not jp2_urls:
            self.logger.warning("No image files found in: %s", batch_url)
            return

        self.logger.info("%s – %d files found", batch_name, len(jp2_urls))

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._download_file, url, batch_url, local_root): url
                for url in jp2_urls
            }
            for _ in tqdm(as_completed(futures), total=len(futures), desc=batch_name):
                pass  # Just for progress bar

    def _find_jp2_files(self, url: str, depth: int = 0, max_depth: int = 6) -> List[str]:
        """Recursively collect image file URLs up to a specified depth."""
        if depth > max_depth:
            return []
        links = self._get_links(url)
        found = []
        for link in links:
            if link.lower().endswith(self.SUPPORTED_EXTS):
                found.append(link)
            elif link.endswith("/"):
                found.extend(self._find_jp2_files(link, depth + 1, max_depth))
        return found

    def _get_links(self, url: str) -> List[str]:
        try:
            r = self.session.get(url, timeout=30)
            r.raise_for_status()
        except Exception as exc:
            self.logger.error("Failed to list %s – %s", url, exc)
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        hrefs = [a.get("href") for a in soup.find_all("a") if a.get("href")]
        return [urljoin(url, h) for h in hrefs if not h.startswith("..")]

    def _download_file(self, file_url: str, batch_url: str, local_root: Path) -> None:
        """Download a single image file, preserving subdirectory structure."""
        rel_path = Path(urlparse(file_url).path.lstrip("/"))
        try:
            parts = rel_path.parts
            idx = parts.index(Path(batch_url).name)
            rel_path = Path(*parts[idx + 1 :])
        except ValueError:
            pass

        local_path = local_root / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and local_path.stat().st_size > 0:
            return

        try:
            with self.session.get(file_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    leave=False,
                    desc=local_path.name,
                ) as bar, local_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
        except Exception as exc:
            self.logger.error("Download failed: %s – %s", file_url, exc)
            self.failed.append({"url": file_url, "error": str(exc)})
            local_path.unlink(missing_ok=True)


# CLI interface
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Mirror JP2/TIFF batches from an HTTP directory")
    p.add_argument("output_dir", help="Local directory to save downloaded files")
    p.add_argument("urls", nargs="+", help="One or more batch root URLs")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel downloads (default: 4)")
    args = p.parse_args()

    JP2Downloader(args.output_dir, max_workers=args.workers).download_all_batches(args.urls)
