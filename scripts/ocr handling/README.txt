# Newspaper OCR Processing Scripts

This directory contains the OCR (Optical Character Recognition) processing scripts used in the Newspaper-OCR-V3 project.

## Scripts Overview

### JP2 Downloader
`newspaper_ocr_download_jp2.py` - Downloads JP2/TIFF image files from newspaper archives.

```
python scripts/newspaper_ocr_download_jp2.py ./downloads https://example.com/batch/url/
```

Features:
- Multi-threaded downloading for efficient retrieval
- Automatic directory traversal and JP2 file discovery
- Resume capability for interrupted downloads
- Progress tracking with detailed logging

### OCR Processing 
`ocr_pipeline.py` - Processes downloaded JP2 files using Google Cloud Vision API and Tesseract.

```
python scripts/ocr/ocr_pipeline.py --config config.yml
```

Features:
- Dual-engine OCR with Google Cloud Vision primary, Tesseract fallback
- Image preprocessing including binarization, contrast enhancement, and deskewing
- Configurable via YAML files
- Checkpointing for long-running jobs

### Verification
`verify_ocr.py` - Validates OCR output and identifies missing or failed files.

```
python scripts/ocr/verify_ocr.py --input-dir ./processed --jp2-dir ./downloads
```

### Organization
`organize_ocr.py` - Organizes OCR output into structured directories by date or page.

```
python scripts/ocr/organize_ocr.py --input-dir ./raw_ocr --output-dir ./organized --mode date
```

## Installation

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up Google Cloud Vision API credentials (if using Vision API):
   ```
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
   ```

## Configuration

See `config_example.yml` for configuration options.