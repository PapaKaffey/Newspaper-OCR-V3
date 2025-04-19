#!/usr/bin/env python3
"""
newspaper_ocr.jp2_downloader
============================
Utility for mirroring Chronicling‑America / OpenONI batch folders
(or any HTTP directory tree) that contain JP2 / TIFF / PNG scans.

Typical usage
-------------
>>> from newspaper_ocr import JP2Downloader
>>> urls = [
...     "https://nebnewspapers.unl.edu/data/batches/batch_nbu_plattsmouth01_ver01/",
... ]
>>> JP2Downloader("./downloads", max_workers=4).download_all_batches(urls)
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
    """Recursively mirror JP2/TIFF/PNG files exposed over an HTTP directory."""

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

        # Robust session with retry/back‑off
        self.session = requests.Session()
        retry_cfg = Retry(
            total=retries,
            backoff_factor=0.8,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry_cfg))
        self.session.mount("http://", HTTPAdapter(max_retries=retry_cfg))

        self.logger = logging.getLogger("jp2_downloader")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------ public

    def download_all_batches(self, batch_urls: Sequence[str]) -> None:
        for idx, url in enumerate(batch_urls, 1):
            self.logger.info("(%d/%d) Batch %s", idx, len(batch_urls), url)
            self._download_batch(url.rstrip("/"))

        if self.failed:
            fail_path = self.base_dir / "failed_downloads.txt"
            with fail_path.open("w", encoding="utf-8") as fh:
                for item in self.failed:
                    fh.write(f"{item['url']}\t{item['error']}\n")
            self.logger.warning(
                "Finished with %d failed files – see %s", len(self.failed), fail_path
            )
        else:
            self.logger.info("All batches downloaded successfully ✨")

    # ---------------------------------------------------------------- internal

    def _download_batch(self, batch_url: str) -> None:
        batch_name = Path(batch_url).name
        local_root = self.base_dir / batch_name

        jp2_urls = self._find_jp2_files(batch_url)
        if not jp2_urls:
            self.logger.warning("No JP2/TIFF files found in %s", batch_url)
            return

        self.logger.info("%s – %d files", batch_name, len(jp2_urls))

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {
                pool.submit(self._download_file, u, batch_url, local_root): u
                for u in jp2_urls
            }
            for _ in tqdm(as_completed(futs), total=len(futs), desc=batch_name):
                pass  # progress only

    # ---------------- directory crawling ------------------------------------

    def _find_jp2_files(
        self, url: str, depth: int = 0, max_depth: int = 6
    ) -> List[str]:
        if depth > max_depth:
            return []
        links = self._links(url)
        out: List[str] = []
        for link in links:
            if link.lower().endswith(self.SUPPORTED_EXTS):
                out.append(link)
            elif link.endswith("/"):
                out.extend(self._find_jp2_files(link, depth + 1, max_depth))
        return out

    def _links(self, url: str) -> List[str]:
        try:
            r = self.session.get(url, timeout=30)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Cannot list %s – %s", url, exc)
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        hrefs = [a.get("href") for a in soup.find_all("a") if a.get("href")]
        return [urljoin(url, h) for h in hrefs if not h.startswith("..")]

    # ---------------- single‑file download -----------------------------------

    def _download_file(self, file_url: str, batch_url: str, local_root: Path) -> None:
        rel_path = Path(urlparse(file_url).path.lstrip("/"))

        # Trim leading path up to batch folder so we mirror the sub‑tree
        try:
            parts = rel_path.parts
            idx = parts.index(Path(batch_url).name)
            rel_path = Path(*parts[idx + 1 :])
        except ValueError:
            pass

        local_path = local_root / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and local_path.stat().st_size > 0:
            return  # already downloaded

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
        except Exception as exc:  # noqa: BLE001
            self.logger.error("%s – %s", file_url, exc)
            self.failed.append({"url": file_url, "error": str(exc)})
            local_path.unlink(missing_ok=True)

# ------------------------------------------------------------------ smoke test
if __name__ == "__main__":  # python -m newspaper_ocr.jp2_downloader DEST URL...
    import argparse

    p = argparse.ArgumentParser(description="Mirror JP2/TIFF batches over HTTP")
    p.add_argument("output_dir")
    p.add_argument("urls", nargs="+")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    JP2Downloader(a.output_dir, max_workers=a.workers).download_all_batches(a.urls)
