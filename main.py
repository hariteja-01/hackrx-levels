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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration (Optimized for Render)
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    MAX_CONTEXT_LENGTH = 5000
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"  # Updated
    MAX_RETRIES = 2
    MAX_CONCURRENT_REQUESTS = 8
if not Config.GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")
# Initialize components
try:
    embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    logger.info(f"Loaded lighter embedding model: {Config.EMBEDDING_MODEL}")
except Exception as e:
    logger.error(f"Embedding model load failed: {str(e)}")
    raise

executor = ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_REQUESTS)
faiss_index = faiss.IndexFlatL2(384)  # Dimension for paraphrase-MiniLM-L3-v2

app = FastAPI(title="Insurance Q&A", version="2.0")

# Data models
class DocumentRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    language: str = "en"

class DocumentResponse(BaseModel):
    answers: List[str]
    rationale: str

# --- Utility Functions ---
@lru_cache(maxsize=32)
def get_embeddings(text: str) -> np.ndarray:
    return embedding_model.encode([text], convert_to_tensor=False)[0]

async def download_pdf(url: str) -> bytes:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(str(url)) as response:
                response.raise_for_status()
                return await response.read()
    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF download failed: {str(e)}")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes)
        return " ".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=422, detail="PDF text extraction failed")

def chunk_text(text: str) -> List[str]:
    return [text[i:i+Config.CHUNK_SIZE] for i in range(0, len(text), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP)]
def build_faiss_index(text_chunks: List[str]):
    try:
        embeddings = np.array([get_embeddings(chunk) for chunk in text_chunks])
        global faiss_index
        faiss_index.reset()
        faiss_index.add(embeddings)
    except Exception as e:
        logger.error(f"FAISS failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Document indexing failed")

@backoff.on_exception(backoff.expo, Exception, max_tries=Config.MAX_RETRIES)
async def generate_with_gemini(prompt: str) -> str:
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': Config.GEMINI_API_KEY  # Updated header
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
            logger.error(f"Gemini API error: {str(e)}")
            raise

# --- API Endpoints ---
@app.post("/hackrx/run", response_model=DocumentResponse)
async def answer_questions(request: DocumentRequest):
    start_time = time.time()
    try:
        # Process PDF
        pdf_bytes = await download_pdf(request.documents)
        text = extract_text_from_pdf(pdf_bytes)
        chunks = chunk_text(text)
        build_faiss_index(chunks)
        
        # Process questions
        answers = await asyncio.gather(*[
            process_question(q, chunks, request.language)
            for q in request.questions
        ])
        
        return DocumentResponse(
            answers=answers,
            rationale=f"Processed {len(request.questions)} questions in {time.time()-start_time:.2f}s"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Processing failed")

async def process_question(question: str, chunks: List[str], language: str) -> str:
    try:
        relevant = retrieve_relevant_chunks(question, chunks)
        context = "\n".join(relevant)[:Config.MAX_CONTEXT_LENGTH]
        prompt = f"Answer in {language} based on:\n{context}\n\nQuestion: {question}\nAnswer:"
        return await generate_with_gemini(prompt)
    except Exception:
        return "Answer unavailable (API limit)"

def retrieve_relevant_chunks(question: str, chunks: List[str], k: int = 2) -> List[str]:
    try:
        query_embedding = get_embeddings(question).reshape(1, -1)
        _, indices = faiss_index.search(query_embedding, k)
        return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
    except Exception:
        return chunks[:k]  # Fallback to first chunks

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": Config.EMBEDDING_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=1  # Critical for Render free tier
    )