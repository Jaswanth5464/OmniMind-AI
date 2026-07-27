# 🧠 OmniMind AI — Offline Local File Intelligence & RAG Search Engine

[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline-green.svg)](https://github.com/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-blue.svg)](https://github.com/facebookresearch/faiss)
[![LLM](https://img.shields.io/badge/LLM-Ollama%20%2F%20TinyLlama-orange.svg)](https://ollama.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

> **OmniMind AI** is an advanced, privacy-first, local search and Retrieval-Augmented Generation (RAG) assistant. It indexes your desktop files, documents, source code, spreadsheets, and images, enabling context-aware natural language search and Q&A **without sending a single byte of data to the cloud**.

---

## 🌟 Key Features in Detail

### 1. 📄 Multi-Format Intelligent File Extraction
- **Supported Formats**: `.pdf`, `.docx`, `.pptx`, `.txt`, `.csv`, `.xlsx`, `.xls`, and all major source code files (`.py`, `.js`, `.ts`, `.java`, `.cpp`, `.h`, `.html`, `.css`, `.json`, `.md`, etc.).
- **Smart Chunking**: Text is extracted, cleaned, and split into semantic chunks with overlap to maintain contextual continuity.
- **Abstractive Summarization**: Utilizes **BART-Large-CNN** to create concise abstract summaries for dense documents.

### 2. 🖼️ Multimodal Vision & Image Understanding
- **Automatic Image Captioning**: Integrated with Salesforce **BLIP (Bootstrapping Language-Image Pre-training)** vision transformer.
- **Natural Language Photo Search**: Scans image files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`) and converts visual content into searchable natural language descriptions (e.g., *"Photo of a blue car parked near a beach"*).

### 3. ⚡ High-Performance FAISS Vector Engine
- **Dense Semantic Embeddings**: Powered by **BAAI/bge-base-en-v1.5** (768-dimensional embeddings) for state-of-the-art semantic match accuracy.
- **Meta FAISS Vector Indexing**: Uses Facebook AI Similarity Search (`IndexFlatIP` with normalized vectors) for sub-millisecond retrieval across tens of thousands of document chunks.
- **Low Memory Footprint**: Runs locally on disk with zero external server dependencies.

### 4. 🤖 Offline Grounded RAG & LLM Synthesis
- **Local Ollama Integration**: Connects locally to **Ollama** (e.g., `tinyllama:1.1b`, `llama3`, `mistral`, or `phi3`).
- **Grounded Answers with Citations**: Compiles retrieved chunks into a targeted prompt so the LLM synthesizes precise answers with exact file paths and source references.
- **Search-Only Fallback**: If Ollama is not installed or running, OmniMind automatically falls back to instant semantic search mode.

### 5. 🔄 Live Watchdog & Incremental Delta Indexing
- **Real-Time Disk Monitoring**: Built-in background directory watcher (`watchdog`) monitors specified folders for changes.
- **Incremental Updates**: Detects **created**, **modified**, and **deleted** files on the fly. Only changed files are re-processed and updated in the FAISS index—eliminating full re-index overhead.
- **Catch-up Sync**: Performs delta scans on startup to catch changes made while the app was offline.

### 6. 🧹 Integrated Duplicate File Finder & Cleaner
- **MD5 Hash Inspection**: Detects exact file duplicates across indexed directories regardless of filename variations.
- **Desktop UI Cleaner**: Built-in interactive window to preview duplicates, review file sizes/locations, and safely clean duplicate files to free disk space.

### 7. 🖥️ Dual Mode Interface (GUI + CLI)
- **Tkinter Desktop GUI**: Modern dark-themed GUI with query search bar, AI answer card, retrieved source list, and **1-click file actions** ("Open File" and "Show in File Explorer").
- **Terminal CLI**: Interactive command-line chat mode for terminal users and SSH environments.

### 8. 🛡️ 100% Privacy & Air-Gapped Security
- **Zero Cloud Leakage**: All embeddings, summaries, vector calculations, and LLM inference occur 100% locally on your machine.
- **Zero Telemetry / No API Keys**: Works completely offline without requiring any subscription, API key, or internet connection.

---
<img width="746" height="462" alt="image" src="https://github.com/user-attachments/assets/a93aa8e8-1c3f-47cd-b643-2f578dbc39f9" />







<img width="1281" height="720" alt="image" src="https://github.com/user-attachments/assets/bba5bf29-4698-4b2e-bb06-a060541be4b3" />






<img width="1333" height="724" alt="image" src="https://github.com/user-attachments/assets/0ea02f6e-7305-471b-bd84-4a030c60896d" />







<img width="766" height="400" alt="image" src="https://github.com/user-attachments/assets/b0ff96dc-6de3-4d06-b087-ea901639264b" />







<img width="1120" height="345" alt="image" src="https://github.com/user-attachments/assets/8c56afb5-7c62-4682-914c-407bda5ed72e" />










<img width="1029" height="455" alt="image" src="https://github.com/user-attachments/assets/45f4c134-3509-451f-b9f4-e90412a5ef84" />





## 🗺️ System Architecture

```
                               ┌─────────────────────────┐
                               │   User Interface        │
                               │  (Tkinter GUI / CLI)    │
                               └──────────┬──────────────┘
                                          │ Question / Query
                                          ▼
                               ┌─────────────────────────┐
                               │       RAG Engine        │
                               │   (app/rag/answer.py)   │
                               └────┬───────────────▲────┘
                 Query Vectors      │               │ Context Chunks
               & Search Request     ▼               │ & Sources
                               ┌─────────────────────────┐
                               │    FAISS Vector DB      │
                               │   (BGE-Base-1.5 Embed)  │
                               └─────────────────────────┘
                                          │
                                          │ Assembled Context + Prompt
                                          ▼
                               ┌─────────────────────────┐
                               │   Local LLM (Ollama)    │
                               │   (TinyLlama / Llama 3) │
                               └─────────────────────────┘
```

---

## 📁 Repository Structure

```
OmniMind-AI/
├── app/
│   ├── chat/
│   │   ├── cli.py               # Terminal interactive chat interface
│   │   └── gui.py               # Tkinter desktop GUI app & duplicate manager
│   ├── core/
│   │   ├── device.py            # Automatic CUDA / MPS / CPU hardware detection
│   │   ├── file_extractors.py   # Multi-format document text parsers
│   │   ├── image_caption.py     # BLIP vision transformer image captioner
│   │   ├── models.py            # BGE embedding & BART summarizer loader
│   │   └── text_utils.py        # Text chunking & normalization utilities
│   ├── indexing/
│   │   ├── builder.py           # Full directory index pipeline builder
│   │   ├── delta.py             # Incremental single-file add/update/delete engine
│   │   └── watcher.py           # Live watchdog background folder monitor
│   ├── rag/
│   │   ├── answer.py            # Prompt builder & context aggregator
│   │   └── llm.py               # Ollama local LLM API connector
│   ├── utils/
│   │   ├── duplicate_finder.py  # MD5 file hash duplicate scanner
│   │   └── misc.py              # File path helper utilities
│   └── config.py                # Global configuration & default parameters
├── download_models.py           # Pre-downloader script for PyTorch models
├── main.py                      # Main entrypoint CLI runner & menu dispatcher
├── Project_Documentation.md     # Deep dive architectural documentation
├── README.md                    # Project documentation & guide
└── requirements.txt             # Python dependencies
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- **Python 3.10+**
- **Git**
- *(Optional)* **NVIDIA GPU with CUDA** for faster embedding & vision inference.
- *(Optional)* **Ollama** installed from [ollama.com](https://ollama.com) for local LLM answers.

### 2. Installation

```bash
# 1. Clone Repository
git clone https://github.com/Jaswanth5464/OmniMind-AI.git
cd OmniMind-AI

# 2. Create Virtual Environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Linux/macOS:
source .venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt
```

### 3. Setup Ollama (Recommended for AI Q&A)
Download and install [Ollama](https://ollama.com), then pull a lightweight model:
```bash
ollama pull tinyllama:1.1b
```
*(You can also use `llama3`, `mistral`, `phi3`, etc. update `app/config.py` if using a different model).*

---

## 🚀 Usage Guide

Launch the main application dispatcher:

```bash
python main.py "C:/Path/To/Your/Documents"
```

### Available Modes:
1. **`index`**: Runs initial complete scan & builds the FAISS vector database for the targeted folder.
2. **`gui`**: Launches the Tkinter Desktop GUI interface with search, AI answers, click-to-open links, and duplicate cleaner.
3. **`cli`**: Launches interactive terminal chat mode.
4. **`watchdog`**: Starts live background directory watching to auto-update index on file modifications.

---

## ⚙️ Model Stack Summary

| Task | Model Used | Reason Selected |
| :--- | :--- | :--- |
| **Embeddings** | `BAAI/bge-base-en-v1.5` | SOTA retrieval accuracy in lightweight 768-dim footprint. |
| **Image Vision** | `Salesforce/blip-image-captioning-base` | Generates descriptive natural language captions for visual search. |
| **Summarization** | `facebook/bart-large-cnn` | Accurate abstractive document summarization. |
| **Local LLM** | `Ollama (tinyllama:1.1b)` | Lightweight, ultra-fast 100% local natural language generator. |
| **Vector DB** | Meta FAISS (`faiss-cpu` / `faiss-gpu`) | Ultra-fast in-memory/on-disk vector search with 0 external service overhead. |

---

## 🛡️ Privacy & Security Assurance

- **No Data Uploads**: Your local files, parsed text, and vector indices never leave your local file system.
- **Air-Gapped Operation**: Once initial HuggingFace models are cached, internet connectivity can be disabled entirely.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check out issues or submit a pull request.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
