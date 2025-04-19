# 📰 Newspaper-OCR-V3
[![OCR Version](https://img.shields.io/badge/OCR-v2.0--vision--primary-blue?style=flat-square)](https://github.com/PapaKaffey/Newspaper-OCR-V3/releases/tag/v2.0-vision-primary)

An optimized OCR pipeline for historical newspapers using Google Vision API and fallback logic.  
Designed for use with collections like **Chronicling America** and **Open ONI**, but adaptable to any scanned newspaper dataset.

---

## ✅ Features

- 🧠 Intelligent preprocessing: deskewing, CLAHE, Sauvola binarization  
- ⚡ GPU acceleration with OpenCV CUDA (auto fallback to CPU)  
- 🧾 Google Cloud Vision OCR (fallback to Tesseract if needed)  
- 🧠 Optional Named Entity Recognition (NER) with Google Cloud NLP  
- 📂 Recursive scan discovery and batch processing  
- 🔄 Skips previously processed files for efficiency  
- ⚙️ Fully YAML-configurable pipeline behavior  

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/PapaKaffey/Newspaper-OCR-V3.git
cd Newspaper-OCR-V3

# Create and activate a virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If using Google Cloud APIs, set your credentials:

```bash
# Windows
set GOOGLE_APPLICATION_CREDENTIALS=path\to\gcp_key.json

# macOS/Linux
export GOOGLE_APPLICATION_CREDENTIALS=path/to/gcp_key.json
```

---

## 🚀 Quickstart

### 1. Download JP2 images from ONI-style batch

```bash
python scripts/download_jp2.py downloads https://nebnewspapers.unl.edu/data/batches/batch_nbu_plattsmouth01_ver01/
```

### 2. Run OCR on downloaded scans

```bash
python scripts/run_ocr.py downloads output --config scripts/config_enhanced.yml
```

### OR use the enhanced GPU-aware pipeline

```bash
python scripts/newspaper_ocr_gpu.py --config scripts/config_enhanced.yml
```

---

## 📂 Output Files

Each processed page generates:

- `output/{filename}.txt` — OCR text  
- `output/{filename}_meta.json` — Metadata (engine used, OCR confidence, fallback logic)  
- `output/{filename}_entities.json` — Named entities (if NER enabled)  
- `output/{filename}_debug/` — Preprocessed images (CLAHE, binarized, deskewed) — zipped if enabled  

---

## ⚙️ Configuration

Edit `scripts/config_enhanced.yml` to control:

- Input/output paths  
- OCR backend & retry behavior  
- Deskewing and binarization  
- Metadata regex extraction  
- Morphology and CLAHE settings  
- Block classification thresholds  

---

## 🛠 Coming Soon

- 📄 PDF output with selectable OCR layers  
- 🌐 Web UI for reviewing OCR quality  
- 🤖 LLM-enhanced summarization (Ollama/LLAMA.cpp)  
- 🧠 Block-level visual classification  

---

## 📄 License

MIT License  
© 2025 [PapaKaffey](https://github.com/PapaKaffey)
```
