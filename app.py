from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import json
from typing import List, Dict
import logging
import time
from math import fabs


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.documents = []
        self.embeddings = []
        self.index = None
        self.current_pdf_name = ""

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF, using OCR for scanned documents if needed."""
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # If no text found, try OCR
                if not page_text.strip():
                    try:
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        page_text = pytesseract.image_to_string(img)
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num}: {e}")
                
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from Ollama."""
        embeddings = []
        for text in texts:
            try:
                # Updated API call for embeddings
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={
                        "model": "nomic-embed-text:latest",  # Updated model name
                        "prompt": text
                    },
                    timeout=30  # Add timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "embedding" in result:
                        embedding = result["embedding"]
                        embeddings.append(embedding)
                    else:
                        logger.error(f"No embedding in response: {result}")
                        embeddings.append([0.0] * 768)
                else:
                    logger.error(f"Failed to get embedding: {response.status_code} - {response.text}")
                    embeddings.append([0.0] * 768)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error getting embedding: {e}")
                embeddings.append([0.0] * 768)
            except Exception as e:
                logger.error(f"Error getting embedding: {e}")
                embeddings.append([0.0] * 768)
        
        return np.array(embeddings, dtype=np.float32)

    def process_pdf(self, pdf_file, filename: str) -> Dict:
        """Process PDF and create vector index."""
        try:
            self.current_pdf_name = filename
            
            # Extract text
            full_text = self.extract_text_from_pdf(pdf_file)
            
            if not full_text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            # Split into chunks
            self.documents = self.text_splitter.split_text(full_text)
            
            # Get embeddings
            self.embeddings = self.get_embeddings(self.documents)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            
            return {
                "success": True,
                "message": f"Successfully processed {filename}",
                "chunks": len(self.documents),
                "total_characters": len(full_text)
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"success": False, "error": str(e)}

    def search_similar_chunks(self, query: str, k: int = 5) -> List[str]:
        """Search for similar chunks using vector similarity."""
        if self.index is None:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.get_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            # Return relevant chunks
            relevant_chunks = []
            for i, idx in enumerate(indices[0]):
                if scores[0][i] > 0.3:  # Similarity threshold
                    relevant_chunks.append(self.documents[idx])
            
            return relevant_chunks
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return self.documents[:k] if self.documents else []

def call_ollama(prompt: str, model: str = "llama3.2:latest") -> str:  # Updated default model
    """Call Ollama API for text generation."""
    try:
        # Updated API call format
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "num_predict": 2000  # Changed from max_tokens to num_predict
                }
            },
            timeout=60  # Increased timeout for longer responses
        )
        
        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                return result["response"]
            else:
                logger.error(f"No response field in result: {result}")
                return f"Error: Unexpected response format from Ollama"
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return f"Error: Failed to get response from Ollama (Status: {response.status_code})"
            
    except requests.exceptions.ConnectionError:
        logger.error("Connection error: Could not connect to Ollama")
        return "Error: Could not connect to Ollama. Make sure Ollama is running on localhost:11434"
    except requests.exceptions.Timeout:
        logger.error("Timeout error: Ollama took too long to respond")
        return "Error: Ollama request timed out"
    except Exception as e:
        logger.error(f"Error calling Ollama: {e}")
        return f"Error: {str(e)}"

def test_ollama_connection():
    """Test if Ollama is responding correctly."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            logger.info(f"Available models: {[model['name'] for model in models.get('models', [])]}")
            return True
        else:
            logger.error(f"Failed to get models: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False

# Initialize processor
pdf_processor = PDFProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large. Maximum size is 50MB'}), 400
        
        # Process PDF
        result = pdf_processor.process_pdf(file, file.filename)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify({'error': result["error"]}), 500
            
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate summary of the PDF."""
    try:
        data = request.get_json()
        summary_type = data.get('type', 'structured')
        
        if not pdf_processor.documents:
            return jsonify({'error': 'No PDF uploaded'}), 400
        
        # Combine first few chunks for summary
        text_for_summary = '\n\n'.join(pdf_processor.documents[:5])  # First 5 chunks
        
        if summary_type == 'structured':
            prompt = f"""Please provide a comprehensive structured summary of the following document:

{text_for_summary}

Format your response with clear sections including:
1. Main Topic/Subject
2. Key Points (3-5 bullet points)
3. Important Details
4. Conclusions/Recommendations (if any)

Keep the summary concise but informative."""
        
        elif summary_type == 'bullet':
            prompt = f"""Create a bullet-point summary of the following document:

{text_for_summary}

Provide 8-10 key bullet points that capture the most important information."""
        
        else:  # executive
            prompt = f"""Create an executive summary of the following document:

{text_for_summary}

Provide a concise executive summary suitable for business professionals, highlighting the main points, key findings, and recommendations in 2-3 paragraphs."""
        
        summary = call_ollama(prompt)
        
        return jsonify({
            'summary': summary,
            'type': summary_type,
            'pdf_name': pdf_processor.current_pdf_name
        })
        
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer questions about the PDF."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if not pdf_processor.documents:
            return jsonify({'error': 'No PDF uploaded'}), 400
        
        # Find relevant chunks
        relevant_chunks = pdf_processor.search_similar_chunks(question, k=3)
        
        if not relevant_chunks:
            return jsonify({'error': 'No relevant information found'}), 400
        
        # Create context from relevant chunks
        context = '\n\n'.join(relevant_chunks)
        
        prompt = f"""Based on the following document content, please answer the question accurately and concisely.

Document Content:
{context}

Question: {question}

Please provide a clear, factual answer based only on the information provided in the document. If the answer cannot be found in the document, please say so."""
        
        answer = call_ollama(prompt)
        
        return jsonify({
            'question': question,
            'answer': answer,
            'pdf_name': pdf_processor.current_pdf_name,
            'sources_used': len(relevant_chunks)
        })
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get current status."""
    ollama_status = test_ollama_connection()
    return jsonify({
        'pdf_loaded': bool(pdf_processor.documents),
        'pdf_name': pdf_processor.current_pdf_name,
        'chunks': len(pdf_processor.documents),
        'ollama_url': OLLAMA_BASE_URL,
        'ollama_connected': ollama_status
    })

if __name__ == '__main__':
    # Test Ollama connection on startup
    print("Testing Ollama connection...")
    if test_ollama_connection():
        print("✓ Ollama connection successful!")
    else:
        print("✗ Ollama connection failed. Make sure Ollama is running.")
    
    # Check if Tesseract is available
    try:
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR is available")
    except:
        logger.warning("Tesseract OCR not found. Scanned PDFs may not work properly.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)