import os
import re
import time
import logging
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status, Request
from pydantic import BaseModel, HttpUrl
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import aiohttp
import asyncio
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import backoff
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Smaller model for memory efficiency
    CHUNK_SIZE = 800  # Reduced chunk size
    CHUNK_OVERLAP = 100  # Reduced overlap
    MAX_CONTEXT_LENGTH = 8000  # Reduced context length
    MAX_CONCURRENT_REQUESTS = 2  # Reduced concurrency
    PDF_DOWNLOAD_TIMEOUT = 10
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    MAX_RETRIES = 2
    MAX_DOCUMENT_SIZE_MB = 5  # Limit document size
    MAX_QUESTIONS = 5  # Limit number of questions per request

# Validate config
if not Config.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Initialize embedding model with memory optimization
try:
    embedding_model = SentenceTransformer(
        Config.EMBEDDING_MODEL,
        device='cpu',
        cache_folder='./model_cache'
    )
    logger.info(f"Embedding model {Config.EMBEDDING_MODEL} loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {str(e)}")
    raise

# Memory-efficient executor
executor = ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_REQUESTS)

# FAISS index with reduced dimensionality
dimension = 384  # Keep original dimension for model compatibility
faiss_index = faiss.IndexFlatL2(dimension)

# FastAPI app with optimized middleware
app = FastAPI(
    title="HackRx 6.0 Document Q&A with Gemini 2.0 Flash",
    description="AI-powered insurance document analysis with RAG",
    version="2.0.1",  # Version bump for changes
    docs_url="/docs",
    redoc_url=None  # Disable redoc to save memory
)

# Data models with size validation
class DocumentRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    language: str = "en"

    @classmethod
    def validate_questions(cls, v):
        if len(v) > Config.MAX_QUESTIONS:
            raise ValueError(f"Maximum {Config.MAX_QUESTIONS} questions allowed")
        return v

class DocumentResponse(BaseModel):
    answers: List[str]
    rationale: str

# Utility functions with memory optimizations
def preprocess_text(text: str) -> str:
    """Clean text with memory efficiency"""
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str) -> List[str]:
    """Generate smaller chunks with less memory usage"""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + Config.CHUNK_SIZE, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_length:
            break
        start = end - Config.CHUNK_OVERLAP
    return chunks

async def download_pdf(url: str) -> bytes:
    """Download PDF with size limit check"""
    try:
        timeout = aiohttp.ClientTimeout(total=Config.PDF_DOWNLOAD_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(str(url)) as response:
                response.raise_for_status()
                
                # Check content type
                if 'application/pdf' not in response.headers.get('Content-Type', ''):
                    raise ValueError("URL does not point to a PDF file")
                
                # Check content length
                content_length = int(response.headers.get('Content-Length', 0))
                if content_length > Config.MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
                    raise ValueError(f"Document exceeds maximum size of {Config.MAX_DOCUMENT_SIZE_MB}MB")
                
                # Stream the response in chunks to avoid memory spikes
                chunks = []
                total_size = 0
                async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                    total_size += len(chunk)
                    if total_size > Config.MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
                        raise ValueError(f"Document exceeds maximum size of {Config.MAX_DOCUMENT_SIZE_MB}MB")
                    chunks.append(chunk)
                
                return b''.join(chunks)
    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download PDF: {str(e)}"
        )

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text with memory monitoring"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
            if len(text_parts) % 10 == 0:  # Periodic check
                current_size = sum(len(p) for p in text_parts)
                if current_size > Config.MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
                    raise ValueError("Extracted text exceeds size limit")
        return preprocess_text("".join(text_parts))
    except Exception as e:
        logger.error(f"PDF text extraction failed: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail="Failed to extract text from PDF document"
        )

@lru_cache(maxsize=8)  # Reduced cache size
def get_embeddings(text: str) -> np.ndarray:
    """Get embeddings with memory optimization"""
    try:
        return embedding_model.encode([text], convert_to_tensor=False, show_progress_bar=False)[0]
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise

def build_faiss_index(text_chunks: List[str]):
    """Build FAISS index with memory efficiency"""
    try:
        # Process embeddings in batches
        batch_size = 10
        embeddings = []
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            batch_embeddings = embedding_model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        # Concatenate and build index
        all_embeddings = np.concatenate(embeddings)
        global faiss_index
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(all_embeddings)
    except Exception as e:
        logger.error(f"FAISS index build failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to build document index"
        )

def retrieve_relevant_chunks(question: str, text_chunks: List[str], k: int = 2) -> List[str]:  # Reduced k
    """Retrieve chunks with memory awareness"""
    try:
        question_embedding = get_embeddings(question).reshape(1, -1)
        _, indices = faiss_index.search(question_embedding, k)
        return [text_chunks[idx] for idx in indices[0] if 0 <= idx < len(text_chunks)]
    except Exception:
        return []

@backoff.on_exception(backoff.expo,
                     (Exception),
                     max_tries=Config.MAX_RETRIES)
async def generate_with_gemini(prompt: str) -> str:
    """Generate with Gemini with memory-efficient handling"""
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': Config.GEMINI_API_KEY
    }
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                Config.GEMINI_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logger.error(f"Gemini API request failed: {str(e)}")
            raise

async def generate_answer(question: str, context: str, language: str) -> str:
    """Generate answer with optimized prompt"""
    try:
        prompt = f"""You are an insurance document expert. Answer the question based strictly on the context.
        Context: {context}
        Question: {question}
        Answer concisely in {language}:"""
        
        return await generate_with_gemini(prompt)
    except Exception as e:
        logger.warning(f"Answer generation failed: {str(e)}")
        return f"Could not generate answer: {str(e)}"

async def process_single_question(question: str, text_chunks: List[str], language: str) -> str:
    """Process question with memory efficiency"""
    try:
        relevant_chunks = retrieve_relevant_chunks(question, text_chunks)
        context = "\n\n".join(relevant_chunks)[:Config.MAX_CONTEXT_LENGTH]
        return await generate_answer(question, context, language)
    except Exception as e:
        logger.error(f"Question processing failed: {str(e)}")
        return "Error processing question"

# API endpoints with memory safeguards
@app.post(
    "/hackrx/run",
    response_model=DocumentResponse,
    responses={
        400: {"description": "Bad request"},
        422: {"description": "Unprocessable entity"},
        500: {"description": "Internal server error"}
    }
)
async def answer_questions(request: DocumentRequest):
    """Main endpoint with memory optimizations"""
    try:
        start_time = time.time()
        
        # Validate request size
        if len(request.questions) > Config.MAX_QUESTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {Config.MAX_QUESTIONS} questions allowed per request"
            )
        
        # Download and process PDF
        pdf_bytes = await download_pdf(request.documents)
        pdf_text = extract_text_from_pdf(pdf_bytes)
        text_chunks = chunk_text(pdf_text)
        
        # Build FAISS index in batches
        build_faiss_index(text_chunks)
        
        # Process questions with limited concurrency
        answers = []
        for question in request.questions:
            answer = await process_single_question(question, text_chunks, request.language)
            answers.append(answer)
        
        # Prepare response
        total_time = time.time() - start_time
        rationale = (
            f"Processed {len(request.questions)} questions in {total_time:.2f}s. "
            f"Used Gemini 2.0 Flash. Analyzed {len(text_chunks)} document chunks."
        )
        
        # Explicit cleanup
        del pdf_bytes
        del pdf_text
        del text_chunks
        
        return DocumentResponse(
            answers=answers,
            rationale=rationale
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Lightweight health check"""
    return {
        "status": "operational",
        "model": "gemini-2.0-flash",
        "rag_enabled": True,
        "memory_optimized": True
    }

# Memory cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    global embedding_model, faiss_index
    del embedding_model
    del faiss_index
    executor.shutdown(wait=False)
    logger.info("Application shutdown complete with memory cleanup")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1,  # Single worker to stay within memory limits
        limit_max_requests=100,  # Prevent memory leaks
        timeout_keep_alive=30  # Shorter keep-alive
    )