# Newspaper-OCR-V3

Modern, GPU‑aware OCR pipeline for historical newspaper scans (JP2, TIFF, PNG). Built for archival batches from [Chronicling America](https://chroniclingamerica.loc.gov/) and [Open ONI](https://open-oni.github.io/), but usable with any structured directory of newspaper images.

## ✅ Features

- 🧠 Smart preprocessing with block detection, deskewing, contrast enhancement
- ⚡ GPU-accelerated filters (OpenCV CUDA) when available
- 🧾 Google Cloud Vision API integration (fallbacks to Tesseract)
- 📑 Named Entity Recognition (optional via Google NLP)
- 📁 Recursive batch downloading from OpenONI-style directories
- 🧪 YAML-configurable processing settings

---

## 📦 Installation

```bash
# clone the repo
$ git clone https://github.com/PapaKaffey/Newspaper-OCR-V3.git
$ cd Newspaper-OCR-V3

# set up Python env
$ python -m venv .venv
$ source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
$ pip install -r requirements.txt
```

If you're using Google Cloud Vision/NLP, ensure your `GOOGLE_APPLICATION_CREDENTIALS` env variable is set, or update `configs/default.yml` with your credentials path.

---

## 🚀 Quickstart

### 1. Download a batch of JP2s

```bash
python scripts/download_jp2.py downloads \
  https://nebnewspapers.unl.edu/data/batches/batch_nbu_plattsmouth01_ver01/
```

### 2. Run OCR on the downloaded scans

```bash
python scripts/run_ocr.py downloads ocr_output \
  --config configs/default.yml
```

---

## 📂 Output

Each processed page produces:

- `ocr_output/{filename}.txt` — OCR’d text
- `ocr_output/{filename}_entities.json` — named entities (optional)
- `ocr_output/debug/` — debug images of processed blocks

---

## 🔧 Configuration

Edit `configs/default.yml` to change:

- batch size
- deskew thresholds
- binarization window
- block filtering heuristics
- confidence thresholds

---

## 🔜 Coming Soon

- Block-level classification (headlines, ads, body)
- Better confidence scoring
- CLI packaging via `setuptools`

---

## 📄 License

MIT — © 2025 [PapaKaffey](https://github.com/PapaKaffey)

