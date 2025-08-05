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
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CONTEXT_LENGTH = 10000
    MAX_CONCURRENT_REQUESTS = 3
    PDF_DOWNLOAD_TIMEOUT = 10
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    MAX_RETRIES = 2

# Validate config
if not Config.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

try:
    embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    logger.info(f"Embedding model {Config.EMBEDDING_MODEL} loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {str(e)}")
    raise

executor = ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_REQUESTS)

# FAISS index
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)

# FastAPI app
app = FastAPI(
    title="HackRx 6.0 Document Q&A with Gemini 2.0 Flash",
    description="AI-powered insurance document analysis with RAG",
    version="2.0.0"
)

# Data models
class DocumentRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    language: str = "en"

class DocumentResponse(BaseModel):
    answers: List[str]
    rationale: str

# Utility functions
def preprocess_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + Config.CHUNK_SIZE, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - Config.CHUNK_OVERLAP
    return chunks

async def download_pdf(url: str) -> bytes:
    try:
        timeout = aiohttp.ClientTimeout(total=Config.PDF_DOWNLOAD_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(str(url)) as response:
                response.raise_for_status()
                if 'application/pdf' not in response.headers.get('Content-Type', ''):
                    raise ValueError("URL does not point to a PDF file")
                return await response.read()
    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download PDF: {str(e)}"
        )

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return preprocess_text("".join(page.get_text() for page in doc))
    except Exception as e:
        logger.error(f"PDF text extraction failed: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail="Failed to extract text from PDF document"
        )

@lru_cache(maxsize=32)
def get_embeddings(text: str) -> np.ndarray:
    return embedding_model.encode([text], convert_to_tensor=False)[0]

def build_faiss_index(text_chunks: List[str]):
    try:
        embeddings = np.array([get_embeddings(chunk) for chunk in text_chunks])
        global faiss_index
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)
    except Exception as e:
        logger.error(f"FAISS index build failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to build document index"
        )

def retrieve_relevant_chunks(question: str, text_chunks: List[str], k: int = 3) -> List[str]:
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
    try:
        relevant_chunks = retrieve_relevant_chunks(question, text_chunks)
        context = "\n\n".join(relevant_chunks)[:Config.MAX_CONTEXT_LENGTH]
        return await generate_answer(question, context, language)
    except Exception as e:
        logger.error(f"Question processing failed: {str(e)}")
        return "Error processing question"

# API endpoints
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
    try:
        start_time = time.time()
        
        # Download and process PDF
        pdf_bytes = await download_pdf(request.documents)
        pdf_text = extract_text_from_pdf(pdf_bytes)
        text_chunks = chunk_text(pdf_text)
        
        # Build FAISS index
        build_faiss_index(text_chunks)
        
        # Process questions
        answers = await asyncio.gather(*[
            process_single_question(q, text_chunks, request.language)
            for q in request.questions
        ])
        
        # Prepare response
        total_time = time.time() - start_time
        rationale = (
            f"Processed {len(request.questions)} questions in {total_time:.2f}s. "
            f"Used Gemini 2.0 Flash. Analyzed {len(text_chunks)} document chunks."
        )
        
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
    return {
        "status": "operational",
        "model": "gemini-2.0-flash",
        "rag_enabled": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")