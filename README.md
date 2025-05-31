## 📌 Features

- **📄 PDF Text Extraction:** Extracts raw text from PDFs using **PyMuPDF (fitz)**.
- **🧾 OCR Fallback:** Uses **Tesseract OCR** for scanned PDFs when direct text extraction fails.
- **🧠 Intelligent Chunking:** Breaks long documents into overlapping text chunks using `RecursiveCharacterTextSplitter` for better context.
- **🔍 FAISS Indexing:** Stores vector embeddings of chunks in a **FAISS** index for fast similarity search.
- **🧬 Ollama Embeddings:** Uses local Ollama models like `nomic-embed-text` to generate embeddings for both PDF chunks and user queries.
- **💬 Context-Aware Q&A:** Answers user questions based on the top-k relevant chunks using Ollama LLMs (e.g., `llama3.2`) via prompt templating.
- **📑 Multi-mode Summarization:** Supports structured, bullet, and executive summaries using the same RAG + LLM pipeline.
- **📶 Ollama Connectivity Checks:** Confirms Ollama is up and models are available.
- **🩺 System Status Reporting:** `/status` endpoint to check loaded PDFs, chunk count, and OCR/model health.

## 🧪 Tech Stack

- **Flask** – RESTful web backend
- **PyMuPDF** – PDF parsing
- **PyTesseract** – OCR fallback for scanned PDFs
- **FAISS** – Vector indexing for semantic retrieval
- **Langchain/TextSplitters** – Chunking and overlap control
- **Ollama** – Local embeddings + LLM responses

## 🤖 Models Used

- **nomic-embed-text** → Generates vector embeddings for chunks and queries
- **llama3.2** → Generates answers and summaries from relevant document chunks
