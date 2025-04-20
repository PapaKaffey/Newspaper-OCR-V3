## 📜 Scripts Overview

### 🗂️ JP2 Downloader  
**File:** `newspaper_ocr_download_jp2.py`  
Downloads JP2/TIFF image files from newspaper archives.

```bash
python scripts/ocr handling/newspaper_ocr_download_jp2.py ./downloads https://example.com/batch/url/
```

✅ Features:
- Multi-threaded downloading for efficient retrieval
- Automatic directory traversal and JP2 file discovery
- Resume capability for interrupted downloads
- Progress tracking with detailed logging

---

### 🧠 OCR Processing  
**File:** `newspaper_ocr_run.py`  
Processes downloaded JP2 files using Google Cloud Vision API (with Tesseract fallback).

```bash
python scripts/ocr handling/newspaper_ocr_run.py --config scripts/ocr handling/config_enhanced.yml
```

✅ Features:
- Dual-engine OCR: Vision API + fallback to Tesseract
- Preprocessing: binarization, contrast enhancement, deskewing
- YAML-configurable pipeline behavior
- Supports checkpointing & batch job control

---

### 🔍 Verification  
**File:** `newspaper_ocr_verif.py`  
Validates OCR output and identifies missing or failed files.

```bash
python scripts/ocr handling/newspaper_ocr_verif.py --input-dir ./processed --jp2-dir ./downloads
```

✅ Outputs:
- Missing page reports
- Percent complete checks
- Coverage summaries

---

### 📁 Organization  
**File:** `newspaper_ocr_organize.py`  
Organizes OCR outputs into structured folders based on issue date or page number.

```bash
python scripts/ocr handling/newspaper_ocr_organize.py --input-dir ./raw_ocr --output-dir ./organized --mode date
```

✅ Modes:
- `date` — organizes by publication date
- `page` — groups by page filename or sequence

---

## ⚙️ Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud Vision API credentials (if using):
```bash
# Windows
set GOOGLE_APPLICATION_CREDENTIALS=path\to\credentials.json

# Mac/Linux
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

---

## ⚙️ Configuration

Edit or duplicate `scripts/ocr handling/config_enhanced.yml` to configure:
- Input/output paths
- OCR engine selection & retries
- Preprocessing options (CLAHE, binarization, deskew)
- Logging & debug flags

---

© 2025 [PapaKaffey](https://github.com/PapaKaffey) — MIT License
