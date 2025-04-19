import os
import re
from pathlib import Path
import logging
import yaml
import cv2
import numpy as np
import pytesseract
from google.cloud import vision
from skimage.filters import threshold_sauvola
from typing import Tuple, Any
import Levenshtein
import zipfile
import fnmatch
import json  # Add missing import
import argparse
import sys
from datetime import datetime

# Define this at the top level before using it
_GPU_OK = cv2.cuda.getCudaEnabledDeviceCount() > 0

# GPU-enabled CLAHE wrapper
def gpu_clahe(gray: np.ndarray, clip: float, grid: tuple) -> np.ndarray:
    if not _GPU_OK or gray is None:
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(gray)
    g = cv2.cuda_GpuMat()
    g.upload(gray)
    clahe = cv2.cuda.createCLAHE(clip, grid)
    result = clahe.apply(g)
    return result.download()


def apply_clahe(gray: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg.get("enable_clahe", False):
        clip = cfg.get("clahe_clip", 2.0)
        grid = tuple(cfg.get("clahe_grid", [8,8]))
        return gpu_clahe(gray, clip, grid)
    return gray

# === Deskew Function ===
def deskew(img: np.ndarray) -> np.ndarray:
    """Deskews an image using minimum area rectangle."""
    # Ensure input is grayscale
    if len(img.shape) == 3:
        gray_for_deskew = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_for_deskew = img

    # Invert image for contour finding (assuming dark text on light background)
    inverted = cv2.bitwise_not(gray_for_deskew)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 5: # Need at least 5 points for minAreaRect
        logging.warning("Not enough points found for deskewing, skipping.")
        return img # Return original image if not enough points

    angle = cv2.minAreaRect(coords)[-1]

    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    logging.debug(f"Detected skew angle: {angle:.2f}")

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Determine border color (e.g., white for light backgrounds)
    corner_pixels = [img[0, 0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    border_color = np.median(corner_pixels).item() if len(img.shape) == 2 else tuple(np.median(np.array(corner_pixels), axis=0).astype(int))

    rotated = cv2.warpAffine(img, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, # Use constant border
                                borderValue=border_color) # Fill with detected border color
    return rotated
# === End Deskew Function ===


def binarize(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """Applies binarization using Sauvola, Adaptive Gaussian, or Otsu."""
    method = cfg.get('binarization_method', 'sauvola') # Default to sauvola

    if method == 'sauvola' and cfg.get("use_sauvola", True):
        win = cfg.get("sauvola_window", 25)
        k = cfg.get("sauvola_k", 0.3)
        thresh = threshold_sauvola(gray, window_size=win, k=k)
        return (gray > thresh).astype(np.uint8) * 255
    elif method == 'adaptive_gaussian':
        block_size = cfg.get('adaptive_block_size', 15)
        C = cfg.get('adaptive_C', 2)
        # Ensure block_size is odd and > 1
        if block_size <= 1: block_size = 3
        if block_size % 2 == 0: block_size += 1
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C
        )
    else: # Fallback to Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu


def ocr_with_vision(bin_img: np.ndarray, client) -> (str, float):
    _, buf = cv2.imencode('.png', bin_img)
    content = buf.tobytes()
    try:
        resp = client.document_text_detection(image=vision.Image(content=content))
        if resp.error.message:
            raise RuntimeError(resp.error.message)
        text = resp.full_text_annotation.text or ""
        conf = (resp.full_text_annotation.pages[0].confidence
                if resp.full_text_annotation.pages else 0.0)
        return text, conf
    except Exception as e:
        logging.warning(f"Vision OCR failed: {e}")
        return None, 0.0


def ocr_with_tesseract(bin_img: np.ndarray, cfg: dict) -> Tuple[str, float]:
    try:
        config_str = cfg.get("tesseract_config", "--oem 1 --psm 6 --dpi 300")
        # Add timeout parameter to prevent hanging
        text = pytesseract.image_to_string(bin_img, config=config_str, lang='eng', timeout=30)
    except Exception as e:
        logging.warning(f"Tesseract OCR failed: {e}")
        text = ''
    return text, 0.0

#You will need this helper function.
def safe_get(cfg: dict, key_path: str, default=None):
    """
    Safely retrieves a value from a nested dictionary using a key path.
    Example: key_path = "preprocessing_params.noise_reduction.median_blur_ksize"
    """
    keys = key_path.split(".")
    current = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def calculate_delta(vision_text: str, tesseract_text: str) -> float:
    """Calculates a delta based on Levenshtein distance."""
    # Normalize whitespace and case for better comparison
    vt = ' '.join((vision_text or '').split()).lower()
    tt = ' '.join((tesseract_text or '').split()).lower()

    if not vt and not tt: return 1.0  # Both empty is perfect match in this context
    if not vt or not tt: return 0.0   # One empty, one not is zero match

    distance = Levenshtein.distance(vt, tt)
    max_len = max(len(vt), len(tt))
    if max_len == 0: return 1.0
    similarity = 1.0 - (distance / max_len)
    return similarity

#This is from previous files, but just make sure this logic is correct
def zip_debug(dir_path: Path):
    """Zips the contents of a directory and deletes the original directory."""
    zip_path = dir_path.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in dir_path.iterdir():
            zf.write(f, arcname=f.name)
    # Remove debug files
    for f in dir_path.iterdir():
        f.unlink()  # Delete all uncompressed files
    dir_path.rmdir()  # delete

def process_image(img_path: Path, cfg: dict, vision_client, output_dir: Path):
    """Loads, preprocesses, OCRs, and saves results for a single image."""
    try:
        # 1. Load and Preprocess Image
        img = cv2.imread(str(img_path))
        if img is None:
            logging.error(f"Failed to load image: {img_path}")
            return

        base = img_path.stem  # Basename of the file

        #Setup debug directory
        parent = img_path.parent
        dbg_dir = parent / f"{base}_debug"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarize the image
        bin_img = binarize(gray, cfg)

        #--- VISION API FIRST ---
        vision_text, vision_conf = ocr_with_vision(bin_img, vision_client)
        source = "Vision API"

        # Tesseract API Triggers (run this if vision is empty)
        trigger_tesseract = False

        if not vision_text:
            trigger_tesseract = True
            logging.info(f"Vision returned no text, triggering Tesseract API for {img_path}")

        # Perform Tesseract call conditionally
        if trigger_tesseract:
            tesseract_text, tesseract_conf = ocr_with_tesseract(bin_img, cfg)  # Run once
            text = tesseract_text  # Set default assuming Vision had nothing
            conf = 0.0
            source = "Tesseract (Vision failed)"

            # If Vision had partial results, compare them
            if vision_text:
                delta = calculate_delta(vision_text, tesseract_text)
                if delta > 0.7:
                    text = vision_text
                    conf = vision_conf
                    source = "Vision"
                logging.info(f"Tesseract Confidence: {tesseract_conf:.2f}, Vision Delta: {delta:.2f}, Chosen Source: {source}")
            else:
                logging.warning(f"Vision failed completely - using Tesseract results")

            logging.debug(f"Tesseract Text: {tesseract_text}")
            logging.debug(f"Vision Text: {vision_text or '[EMPTY]'}")

        # Did NOT need to run Tesseract
        else:
            text = vision_text
            conf = vision_conf
            source = "Vision API"
            logging.debug(f"Vision Text: {vision_text}")

        # 4. Save Outputs (no Date-Organized Paths)
        txt_path = output_dir / f"{base}.txt"
        json_path = output_dir / f"{base}_meta.json"

        txt_path.write_text(text, encoding='utf-8')
        json_path.write_text(json.dumps( {'text': text, 'confidence': conf, 'source': source }, indent=2), encoding='utf-8')
        logging.info(f"Saved text to {txt_path} (Source: {source}, Conf: {conf:.2f})")
        zip_dbg = safe_get(cfg, 'zip_debug_images', False) # Check debug toggle

        if zip_dbg and dbg_dir.exists() and dbg_dir.is_dir():
            zip_debug(dbg_dir)  # This will zip and remove the debug directory

    except Exception as e:
        logging.exception(f"Error processing {img_path}: {e}")
        problem_dir = Path(safe_get(cfg, "problem_images_dir", "./problem_images"))
        problem_dir.mkdir(parents=True, exist_ok=True)
        problem_path = problem_dir / img_path.name
        img_path.rename(problem_path)
        logging.error(f"Moved problematic image to {problem_path}")

def discover_images(base_dir: Path, cfg: dict) -> list:
    """Find images to process with all filtering rules applied"""
    
    print(f"Scanning for images in: {base_dir}")
    print(f"Directory exists: {base_dir.exists()}")
    
    if not base_dir.exists():
        print(f"ERROR: Base directory {base_dir} does not exist!")
        return []
        
    # Get configuration settings
    exts = cfg.get('supported_extensions', ['.jp2'])
    print(f"Looking for files with extensions: {exts}")
    
    # Make sure extensions have dots
    exts = [ext if ext.startswith('.') else f'.{ext}' for ext in exts]
    
    exclude_exts = cfg.get('file_discovery', {}).get('exclude_extensions', [])
    useless_patterns = cfg.get('file_discovery', {}).get('useless_dir_patterns', [])
    metadata_patterns = cfg.get('metadata_file_patterns', [])

    logging.info(f"Starting image discovery in {base_dir}")
    logging.info(f"Supported extensions: {exts}")
    logging.info(f"Excluded extensions: {exclude_exts}")
    logging.info(f"Useless dir patterns: {useless_patterns}")
    logging.info(f"Metadata file patterns: {metadata_patterns}")

    # Build a list of all candidate files
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)

        # Skip directories matching useless patterns
        if any(root_path.match(pattern) for pattern in useless_patterns):
            logging.debug(f"Skipping directory {root_path} due to useless pattern.")
            dirs.clear()  # Don't descend into this directory
            continue

        for file in files:
            file_path = root_path / file

            # Skip files with excluded extensions
            if any(file.lower().endswith(ext) for ext in exclude_exts):
                logging.debug(f"Skipping file {file} due to excluded extension.")
                continue

            # Skip files not matching our target extensions
            if not any(file.lower().endswith(ext.lower()) for ext in exts):
                print(f"Skipping file {file} - extension not in {exts}")
                logging.debug(f"Skipping file {file} - extension not in {exts}")
                continue

            # Skip metadata files
            if any(fnmatch.fnmatch(file, pattern) for pattern in metadata_patterns):
                logging.debug(f"Skipping file {file} - metadata file pattern.")
                continue

            all_files.append(file_path)

    print(f"Found {len(all_files)} total candidate files")
    
    # Filter out files that have already been processed
    output_dir = Path(cfg['output_dir'])
    not_processed = []
    for img_path in all_files:
        txt_path = output_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            not_processed.append(img_path)
        else:
            print(f"Skipping already processed file: {img_path}")
    
    print(f"Found {len(not_processed)} files that need processing")
    return not_processed

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Simple OCR Pipeline")
    parser.add_argument('--config', required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"ERROR: Failed to load config file: {e}")
        sys.exit(1)

    # Setup logging
    log_level = cfg.get('log_level', 'INFO').upper()
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get base directory (allow override for testing)
    base_dir_override = os.environ.get('OCR_OVERRIDE_BASE_DIR')
    if base_dir_override:
        base_dir = Path(base_dir_override)
        logging.warning(f"Overriding base_dir with: {base_dir}")
    else:
        base_dir = Path(cfg['base_dir'])

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    # Set Google credentials
    credentials_path = cfg.get('credentials_path')
    if credentials_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        logging.info(f"Using Google credentials: {credentials_path}")
    else:
        logging.warning("Google credentials path not set in config.")

    # Initialize Vision client (handle potential errors)
    try:
        vision_client = vision.ImageAnnotatorClient()
        try:
            # Quick test of Vision API connectivity
            test_response = vision_client.label_detection(image=vision.Image(content=b'test'))
            logging.info("Vision API connectivity test successful")
        except Exception as e:
            logging.error(f"Vision API connectivity test failed: {e}")
    except Exception as e:
        logging.error(f"Failed to initialize Google Vision client: {e}")
        logging.error("Ensure credentials are set correctly and network is available.")
        sys.exit(1)

    print("=== OCR PIPELINE STARTING ===")
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")

    # --- Process all images ---
    images_to_process = discover_images(base_dir, cfg)
    if not images_to_process:
        logging.error(f"No images found in {base_dir} matching criteria. Check your base_dir and file filters.")
        print(f"ERROR: No images found in {base_dir}. Check paths and filters.")
        sys.exit(1)

    logging.info(f"Found {len(images_to_process)} images to process")
    # Test directory logic

    for img_path in images_to_process: #Iterate through each image
        logging.info(f"Processing {img_path}")
        process_image(img_path, cfg, vision_client, output_dir)


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser(description="OCR Pipeline for newspaper images")
    parser.add_argument("--config", default="config_enhanced.yml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        main()  # Call your main function
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)