#!/usr/bin/env python3
"""
newspaper_ocr.ocr_pipeline
==========================
GPU‑aware, Google‑Vision‑optional OCR pipeline for newspaper scans.

The public entry point is :pyfunc:`run_ocr`.  Use it from Python **or**
the tiny CLI wrapper in ``scripts/run_ocr.py``.

Minimal example
---------------
>>> from newspaper_ocr import run_ocr
>>> run_ocr("./scans", "./ocr_out")
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytesseract
import yaml
from skimage.filters import threshold_sauvola
from skimage import measure

# --------------------------------------------------------------------------- optional Google extras
try:
    from google.cloud import vision
    from google.cloud import language_v1
    from google.api_core.exceptions import GoogleAPIError, RetryError, ResourceExhausted

    _GCLOUD_OK = True
except Exception:  # pragma: no cover
    vision = language_v1 = None  # type: ignore
    GoogleAPIError = RetryError = ResourceExhausted = Exception  # type: ignore
    _GCLOUD_OK = False

# --------------------------------------------------------------------------- CUDA helpers
_GPU_OK = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, "cuda") else False


def _gpu(img, fn_cpu, fn_cuda, *args, **kw):
    if not _GPU_OK:
        return fn_cpu(img, *args, **kw)
    g = cv2.cuda_GpuMat()
    g.upload(img)
    out = fn_cuda(g, *args, **kw)
    return out.download()


def _clahe(gray, clip=2.0):
    return _gpu(
        gray,
        lambda img, clip: cv2.createCLAHE(clip, (8, 8)).apply(img),
        lambda g, clip: cv2.cuda.createCLAHE(clip, (8, 8)).apply(g),
        clip,
    )


# --------------------------------------------------------------------------- defaults (editable via YAML)
_DEFAULT_CFG: Dict[str, Any] = {
    "supported_extensions": [".jp2", ".tif", ".tiff", ".png"],
    "batch_size": 4,
    "api_rate_limit_delay": 0.0,
    "enable_deskewing": True,
    "max_skew_angle": 6.0,
    "skew_threshold": 0.4,
    "min_confidence": 0.0,
    "preprocessing_params": {
        "default": {
            "clahe": True,
            "blur": True,
            "blur_kernel": (3, 3),
            "sauvola_window": 35,
            "sauvola_k": 0.2,
            "closing": True,
            "closing_kernel_size": (2, 2),
        }
    },
}


# --------------------------------------------------------------------------- small helpers
def _load_cfg(path: str | Path | None) -> Dict[str, Any]:
    if path and Path(path).is_file():
        with open(path, "r", encoding="utf-8") as fh:
            custom = yaml.safe_load(fh) or {}
        return {**_DEFAULT_CFG, **custom}
    return _DEFAULT_CFG.copy()


def _log_setup(log_dir: Path | str):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("newspaper_ocr")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_dir / "ocr.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    return logger


def _discover(folder: Path, exts: List[str]) -> List[Path]:
    out: List[Path] = []
    for ext in exts:
        out.extend(folder.rglob(f"*{ext}"))
    return sorted(out)


# --------------------------------------------------------------------------- rudimentary deskew
def _deskew(img: np.ndarray, max_angle: float, threshold: float) -> Tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return img, 0.0
    angles = [float(np.degrees(theta) - 90) for rho, theta in lines[:, 0]]
    angles = [a for a in angles if abs(a) <= max_angle]
    if not angles:
        return img, 0.0
    skew = float(np.median(angles))
    if abs(skew) < threshold:
        return img, skew
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC), skew


# --------------------------------------------------------------------------- block finding
def _find_blocks(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe(gray, 2.0)
    th = threshold_sauvola(enhanced, 51, 0.1)
    bin_img = (enhanced > th).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)
    label = measure.label(dil)
    regions = measure.regionprops(label)
    h, w = img.shape[:2]
    out = []
    for r in regions:
        if r.area < 300 or r.area > w * h * 0.8:
            continue
        minr, minc, maxr, maxc = r.bbox
        out.append((minc, minr, maxc, maxr))
    return out or [(0, 0, w, h)]  # fallback: whole page


# --------------------------------------------------------------------------- OCR helpers
def _tess(img: np.ndarray):
    return pytesseract.image_to_string(img, lang="eng"), 0.0


def _vision(img_bytes: bytes, client, logger):
    try:
        resp = client.document_text_detection(image=vision.Image(content=img_bytes))
        if resp.error.message:
            raise GoogleAPIError(resp.error.message)  # type: ignore
        txt = resp.full_text_annotation.text  # type: ignore
        conf = resp.full_text_annotation.pages[0].confidence if resp.full_text_annotation.pages else 0.0  # type: ignore
        return txt, conf
    except Exception as e:  # noqa: BLE001
        logger.warning("Vision OCR failed: %s", e)
        return "", 0.0


# --------------------------------------------------------------------------- per‑page processor
def _process(path: Path, cfg: Dict[str, Any], vision_client, logger):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Cannot open %s", path)
        return False, 0.0

    if cfg["enable_deskewing"]:
        img, ang = _deskew(img, cfg["max_skew_angle"], cfg["skew_threshold"])
        logger.debug("%s skew %.2f°", path.name, ang)

    blocks = _find_blocks(img)
    merged, confs = [], []

    for i, (x1, y1, x2, y2) in enumerate(blocks):
        roi = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = _clahe(gray, 2.0)
        th = threshold_sauvola(gray, 35, 0.2)
        proc = (gray > th).astype(np.uint8) * 255

        txt, conf = ("", 0.0)
        if _GCLOUD_OK and vision_client:
            _, buf = cv2.imencode(".png", proc)
            txt, conf = _vision(buf.tobytes(), vision_client, logger)
        if not txt:
            txt, conf = _tess(proc)

        if conf >= cfg["min_confidence"]:
            merged.append(txt.strip())
            confs.append(conf)

    out = "\n\n".join(merged)
    if out:
        out_dir = Path(cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{path.stem}.txt").write_text(out, encoding="utf-8")

    return True, float(np.mean(confs)) if confs else 0.0


# --------------------------------------------------------------------------- public API
def run_ocr(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    log_dir: str | Path = "logs",
):
    cfg = _load_cfg(config_path)
    cfg["output_dir"] = str(Path(output_dir).expanduser().resolve())

    log = _log_setup(log_dir)
    log.info("CUDA available: %s | Google Vision: %s", _GPU_OK, _GCLOUD_OK)

    if cfg.get("credentials_path"):
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", cfg["credentials_path"])

    vision_client = vision.ImageAnnotatorClient() if _GCLOUD_OK else None

    files = _discover(Path(input_dir).expanduser(), cfg["supported_extensions"])
    if not files:
        log.error("No images found in %s", input_dir)
        return

    ok = fail = 0
    confs: List[float] = []
    start = time.time()

    for chunk in range(0, len(files), cfg["batch_size"]):
        batch = files[chunk : chunk + cfg["batch_size"]]
        with ThreadPoolExecutor(max_workers=cfg["batch_size"]) as pool:
            futs = {pool.submit(_process, p, cfg, vision_client, log): p for p in batch}
            for f in as_completed(futs):
                success, conf = f.result()
                (ok if success else fail) += 1
                if success:
                    confs.append(conf)
                time.sleep(cfg["api_rate_limit_delay"])

        log.info(
            "Progress %d/%d | ok:%d fail:%d | mean conf %.2f",
            ok + fail,
            len(files),
            ok,
            fail,
            float(np.mean(confs)) if confs else 0.0,
        )

    log.info("Done in %.1f min", (time.time() - start) / 60)


# --------------------------------------------------------------------------- CLI helper
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run OCR on a folder of scans")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--config")
    ap.add_argument("--log-dir", default="logs")
    a = ap.parse_args()
    run_ocr(a.input_dir, a.output_dir, a.config, a.log_dir)
