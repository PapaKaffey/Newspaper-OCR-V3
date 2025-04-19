#!/usr/bin/env python3
"""
newspaper_ocr.ocr_pipeline
==========================
Public version of the OCR pipeline for scanned newspaper images.
Supports GPU acceleration, Google Cloud Vision, and Tesseract fallback.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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

# Optional Google Cloud support
try:
    from google.cloud import vision
    from google.cloud import language_v1
    from google.api_core.exceptions import GoogleAPIError, RetryError, ResourceExhausted
    _GCLOUD_OK = True
except ImportError:
    vision = language_v1 = None  # type: ignore
    GoogleAPIError = RetryError = ResourceExhausted = Exception  # type: ignore
    _GCLOUD_OK = False

_GPU_OK = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, "cuda") else False

# -------------------- Utility functions --------------------
def _gpu(img, fn_cpu, fn_cuda, *args, **kwargs):
    if not _GPU_OK:
        return fn_cpu(img, *args, **kwargs)
    g = cv2.cuda_GpuMat()
    g.upload(img)
    out = fn_cuda(g, *args, **kwargs)
    return out.download()

def _clahe(gray, clip=2.0):
    return _gpu(
        gray,
        lambda img, clip: cv2.createCLAHE(clip, (8, 8)).apply(img),
        lambda g, clip: cv2.cuda.createCLAHE(clip, (8, 8)).apply(g),
        clip,
    )

# -------------------- Config and Logging --------------------
_DEFAULT_CFG = {
    "supported_extensions": [".jp2", ".tif", ".tiff", ".png"],
    "batch_size": 4,
    "api_rate_limit_delay": 0.1,
    "enable_deskewing": True,
    "max_skew_angle": 6.0,
    "skew_threshold": 0.4,
    "min_confidence": 0.5,
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
    "output_dir": "ocr_output",
}

def _load_cfg(path):
    if path and Path(path).is_file():
        with open(path, "r", encoding="utf-8") as f:
            return {**_DEFAULT_CFG, **yaml.safe_load(f)}
    return _DEFAULT_CFG.copy()

def _setup_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ocr_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("ocr_pipeline")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    return logger

def _discover_images(folder: Path, extensions: List[str]) -> List[Path]:
    return sorted(p for ext in extensions for p in folder.rglob(f"*{ext}"))

# -------------------- Image Processing --------------------
def _deskew(img, max_angle, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return img, 0.0
    angles = [np.degrees(theta) - 90 for rho, theta in lines[:, 0] if abs(np.degrees(theta) - 90) <= max_angle]
    if not angles:
        return img, 0.0
    skew = float(np.median(angles))
    if abs(skew) < threshold:
        return img, skew
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC), skew

def _find_blocks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = _clahe(gray, 2.0)
    th = threshold_sauvola(clahe, 51, 0.1)
    binary = (clahe > th).astype(np.uint8) * 255
    dilated = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    label_img = measure.label(dilated)
    regions = measure.regionprops(label_img)
    h, w = img.shape[:2]
    return [(minc, minr, maxc, maxr) for r in regions if 300 < r.area < h * w * 0.8] or [(0, 0, w, h)]

def _ocr_vision(img_bytes, client, logger):
    try:
        response = client.document_text_detection(image=vision.Image(content=img_bytes))
        if response.error.message:
            raise GoogleAPIError(response.error.message)
        txt = response.full_text_annotation.text
        conf = response.full_text_annotation.pages[0].confidence if response.full_text_annotation.pages else 0.0
        return txt.strip(), conf
    except Exception as e:
        logger.warning(f"Vision API failed: {e}")
        return "", 0.0

def _ocr_tesseract(img):
    return pytesseract.image_to_string(img, lang="eng"), 0.0

def _process(path: Path, cfg: Dict[str, Any], vision_client, logger):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Cannot open {path}")
        return False, 0.0

    if cfg["enable_deskewing"]:
        img, angle = _deskew(img, cfg["max_skew_angle"], cfg["skew_threshold"])
        logger.debug(f"{path.name} deskewed {angle:.2f}°")

    blocks = _find_blocks(img)
    merged, confs = [], []

    for i, (x1, y1, x2, y2) in enumerate(blocks):
        roi = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if cfg["preprocessing_params"]["default"].get("clahe"):
            gray = _clahe(gray)
        th = threshold_sauvola(gray, 35, 0.2)
        proc = (gray > th).astype(np.uint8) * 255

        _, buf = cv2.imencode(".png", proc)
        img_bytes = buf.tobytes()

        if _GCLOUD_OK and vision_client:
            txt, conf = _ocr_vision(img_bytes, vision_client, logger)
        else:
            txt, conf = _ocr_tesseract(proc)

        if txt and conf >= cfg["min_confidence"]:
            merged.append(txt)
            confs.append(conf)

    if merged:
        out_dir = Path(cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{path.stem}.txt").write_text("\n\n".join(merged), encoding="utf-8")
    return True, float(np.mean(confs)) if confs else 0.0

# -------------------- Entry Point --------------------
def run_ocr(input_dir: str, output_dir: str, config_path: str | None = None, log_dir: str = "logs"):
    start_time = time.time()
    cfg = _load_cfg(config_path)
    cfg["output_dir"] = output_dir
    logger = _setup_logger(log_dir)
    logger.info("CUDA available: %s | Google Cloud Vision: %s", _GPU_OK, _GCLOUD_OK)

    if cfg.get("credentials_path"):
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", cfg["credentials_path"])

    vision_client = vision.ImageAnnotatorClient() if _GCLOUD_OK else None
    files = _discover_images(Path(input_dir), cfg["supported_extensions"])
    if not files:
        logger.error("No input images found.")
        return

    ok, fail, confs = 0, 0, []
    for i in range(0, len(files), cfg["batch_size"]):
        batch = files[i:i+cfg["batch_size"]]
        with ThreadPoolExecutor(max_workers=cfg["batch_size"]) as pool:
            futures = [pool.submit(_process, f, cfg, vision_client, logger) for f in batch]
            for fut in as_completed(futures):
                success, conf = fut.result()
                if success:
                    ok += 1
                    confs.append(conf)
                else:
                    fail += 1
                time.sleep(cfg["api_rate_limit_delay"])
        logger.info("Progress %d/%d | ok %d | fail %d | avg_conf %.2f", ok+fail, len(files), ok, fail, np.mean(confs) if confs else 0.0)

    logger.info("OCR completed in %.1f minutes", (time.time() - start_time) / 60)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run OCR on a batch of images")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--config")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()
    run_ocr(args.input_dir, args.output_dir, args.config, args.log_dir)
