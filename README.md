# 📰 Newspaper-OCR-V3

A modern OCR pipeline for historical newspaper scans with GPU acceleration, smart preprocessing, and flexible backends. Designed for use with collections like [Chronicling America](https://chroniclingamerica.loc.gov/) and [Open ONI](https://open-oni.github.io/), but adaptable to any scanned newspaper dataset.

---

## ✅ Features

- 🧠 Intelligent preprocessing: deskewing, CLAHE, Sauvola binarization
- ⚡ GPU acceleration with OpenCV CUDA (auto fallback to CPU)
- 🧾 Google Cloud Vision OCR (fallbacks to Tesseract if needed)
- 🧠 Optional Named Entity Recognition (NER) with Google Cloud NLP
- 📂 Recursive scan discovery and batch processing
- 🔄 Skips previously processed files for efficiency
- ⚙️ Fully YAML-configurable pipeline behavior

---

## 📦 Installation

```bash
# Clone the repository
$ git clone https://github.com/PapaKaffey/Newspaper-OCR-V3.git
$ cd Newspaper-OCR-V3

# Create and activate a virtual environment
$ python -m venv venv
# On Windows
$ venv\Scripts\activate
# On macOS/Linux
$ source venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt
```

If using Google Cloud APIs, set your credentials path:

```bash
# Windows
set GOOGLE_APPLICATION_CREDENTIALS=path\to\gcp_key.json

# macOS/Linux
export GOOGLE_APPLICATION_CREDENTIALS=path/to/gcp_key.json
```

---

## 🚀 Quickstart

### 1. Download JP2 images from an ONI-style batch

```bash
python scripts/download_jp2.py downloads https://nebnewspapers.unl.edu/data/batches/batch_nbu_plattsmouth01_ver01/
```

### 2. Run OCR on downloaded scans

```bash
python scripts/run_ocr.py downloads output --config Scripts/config_enhanced.yml
```

### Or run the enhanced GPU-aware pipeline

```bash
python Scripts/newspaper_ocr_gpu.py --config Scripts/config_enhanced.yml
```

---

## 📂 Output Files

Each processed page generates:

- `output/{filename}.txt` — OCR text
- `output/{filename}_meta.json` — Metadata (date, paper, issue, etc.)
- `output/{filename}_entities.json` — Named entities (if enabled)
- `output/{filename}_debug/` — Preprocessed block images

---

## ⚙️ Configuration

Edit `Scripts/config_enhanced.yml` to adjust:

- Input/output paths
- OCR backends and retry behavior
- Deskewing and binarization options
- Metadata regex patterns
- Block classification thresholds

---

## 🛠 Coming Soon

- PDF output with selectable OCR layers
- Web UI for visual review of OCR quality
- LLM-enhanced summarization (Ollama integration)
- Block-level visual classification

---

## 📄 License

MIT License  
© 2025 [PapaKaffey](https://github.com/PapaKaffey)
