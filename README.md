#  PDF-Summarizer-QA-Web-App-using-Ollama

A Flask-based web application that lets you **upload PDFs**, **generate intelligent summaries**, and **ask context-aware questions** using local **Ollama** models with embeddings and LLMs. It uses **FAISS** for semantic search and supports scanned PDFs via **OCR** fallback.

---


## Features

1. **PDF Text Extraction** - Extracts text from PDFs using PyMuPDF
2. **OCR Fallback** - Uses Tesseract OCR for scanned PDFs
3. **Intelligent Chunking** - Breaks documents into overlapping chunks
4. **FAISS Indexing** - Vector embeddings for fast similarity search
5. **Ollama Embeddings** - Local embedding generation with nomic-embed-text
6. **Context-Aware Q&A** - Answers questions using llama3.2 LLM
7. **Multi-mode Summarization** - Structured, bullet, and executive summaries
8. **System Status Reporting** - Check loaded PDFs and system health

## Tech Stack

- **Flask** - Web backend
- **PyMuPDF** - PDF parsing
- **PyTesseract** - OCR processing
- **FAISS** - Vector indexing
- **Langchain** - Text splitting and chunking
- **Ollama** - Local LLM and embeddings

## API Endpoints

- `/status` - Check system status and loaded PDFs
- Upload and process PDFs through the web interface
- Ask questions about uploaded documents
- Generate summaries in different formats

##  Models Used

- `nomic-embed-text` → Generates vector embeddings for chunks and queries
- `llama3.2` → Generates answers and summaries from relevant document chunks

## Prerequisites

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Ollama](https://ollama.com) running locally at http://localhost:11434

### Install Required Ollama Models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/shiuli-19/PDF-Summarizer-QA-Web-App-using-Ollama
cd PDF_chatbot
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama Service

Ensure Ollama is running in the background on port 11434.

### 5. Run the Application

```bash
python run.py
```

The application will start on `http://localhost:5000`

## Verification Steps

1. **Start Ollama service**
   ```bash
   ollama serve
   ```

2. **Run the application**
   ```bash
   python run.py
   ```

3. **Open browser**
   Navigate to `http://localhost:5000`

4. **Test functionality**
   - Upload a PDF file
   - Ask questions about the document
   - Generate summaries

## Troubleshooting

### Check Ollama Status
```bash
ollama list
```

### Verify Models are Downloaded
```bash
ollama show nomic-embed-text
ollama show llama3.2
```

### Check Application Status
Visit `http://localhost:5000/status` to see system health and loaded PDFs.
