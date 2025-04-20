# 📰 Newspaper-OCR-V3
[![OCR Version](https://img.shields.io/badge/OCR-v2.1--semantic--chunked-brightgreen?style=flat-square)](https://github.com/PapaKaffey/Newspaper-OCR-V3/releases/tag/v2.1)

An AI-powered OCR pipeline optimized for historical newspapers using Google Vision API, fallback logic, and vector search.  
Designed for collections like **Chronicling America** and **Open ONI**, now upgraded with FAISS, Ollama, and RAG capabilities.

---

## ✅ Features

- 🧠 Intelligent preprocessing: deskewing, CLAHE, Sauvola binarization  
- ⚡ GPU acceleration with OpenCV CUDA (auto fallback to CPU)  
- 🧾 Google Cloud Vision OCR (fallback to Tesseract)  
- 🔍 Article chunking & semantic search using FAISS  
- 🤖 LLM Q&A using Ollama & local Retrieval-Augmented Generation  
- 📂 Recursive scan discovery & batch processing  
- 🔄 Skips already processed files (re-entrant pipeline)  
- ⚙️ YAML-configurable workflow  

---

## 📦 Installation

git clone https://github.com/PapaKaffey/Newspaper-OCR-V3.git
cd Newspaper-OCR-V3

python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt


If using Google Cloud Vision OCR:


# Windows
set GOOGLE_APPLICATION_CREDENTIALS=path\to\gcp_key.json

# macOS/Linux
export GOOGLE_APPLICATION_CREDENTIALS=path/to/gcp_key.json


---

## 🚀 Quickstart

### 1. Download JP2 images from archive


python scripts/ocr handling/newspaper_ocr_download_jp2.py downloads https://nebnewspapers.unl.edu/data/batches/batch_nbu_plattsmouth01_ver01/


### 2. Run OCR on downloaded scans

python scripts/ocr handling/newspaper_ocr_run.py --config scripts/ocr handling/config_enhanced.yml


### 3. Organize OCR files by date or page


python scripts/ocr handling/newspaper_ocr_organize.py --input-dir ./processed --output-dir ./organized --mode date

---

## 🔹 Chunk + Embed with FAISS Index


python scripts/search/chunked_embed_and_search.py

Creates:
- `vector_index/chunked_faiss.index`
- `chunked_faiss_metadata.json`



## 🔹 Ask LLM Questions Using RAG


python scripts/search/rag_query_ollama_chunked.py --query "Where did Weckbach bury the gold?" --model deepcoder:1.5b


---

## 📂 Output Structure

- `output/*.txt` → OCR text  
- `*_meta.json` → OCR engine, fallback, confidence  
- `*_enriched.json` → NER + paragraph chunks  
- `vector_index/` → FAISS vector store + metadata  

---

## ⚙️ Configuration

Edit `scripts/ocr handling/config_enhanced.yml` to control:

- Input/output paths  
- OCR engine fallback  
- Preprocessing flags (CLAHE, Sauvola, deskew)  
- Entity recognition options  
- Debug and logging options  

---

## 📄 License

MIT License  
© 2025 [PapaKaffey](https://github.com/PapaKaffey)
