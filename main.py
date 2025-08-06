import os
import re
import time
import logging
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, HttpUrl
from sentence_transformers import SentenceTransformer
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import aiohttp
import asyncio
import atexit

# Load environment variables
load_dotenv()

# Configuration
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"  # Smaller, faster model
    CHUNK_SIZE = 500          # Reduced for lower memory
    CHUNK_OVERLAP = 100
    MAX_CONTEXT_LENGTH = 3000 # Reduced context to save memory
    MAX_CONCURRENT_REQUESTS = 2  # Limit concurrency
    PDF_DOWNLOAD_TIMEOUT = 15
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    MAX_RETRIES = 2

# Validate config
if not Config.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Initialize FastAPI
app = FastAPI(
    title="HackRx 6.0 Document Q&A with Gemini 1.5 Flash",
    description="AI-powered insurance document analysis with RAG",
    version="2.0.1"
)

# Global variables (lazy-loaded)
embedding_model = None
executor = None

def load_model():
    global embedding_model, executor
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logging.info(f"Embedding model '{Config.EMBEDDING_MODEL}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {str(e)}")
            raise

    # Reuse or create executor
    global executor
    if executor is None:
        executor = asyncio.get_event_loop().create_task(asyncio.sleep(0))  # placeholder

# Lazy load model on first request
@app.on_event("startup")
async def startup_event():
    # Warm-up: load model during startup (optional, can be moved to first request)
    load_model()

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
        chunks.append(preprocess_text(chunk))
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
                content_type = response.headers.get('Content-Type', '').lower()
                if 'pdf' not in content_type and 'application/pdf' not in content_type:
                    raise ValueError("URL does not point to a PDF file")
                return await response.read()
    except Exception as e:
        logging.error(f"PDF download failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return preprocess_text(text)
    except Exception as e:
        logging.error(f"PDF text extraction failed: {str(e)}")
        raise HTTPException(status_code=422, detail="Failed to extract text from PDF")

def get_embeddings(texts: List[str]) -> np.ndarray:
    load_model()  # Ensure model is loaded
    return embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

async def generate_with_gemini(prompt: str) -> str:
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512
        }
    }

    url = f"{Config.GEMINI_URL}?key={Config.GEMINI_API_KEY}"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    logging.error(f"Gemini API error {response.status}: {text}")
                    return "I couldn't generate an answer due to an internal error."
                result = await response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logging.error(f"Gemini API request failed: {str(e)}")
            return "Error: Could not reach AI model."

async def process_single_question(question: str, chunks: List[str], language: str) -> str:
    try:
        # Create embeddings for chunks
        embeddings = get_embeddings(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Search
        q_embed = get_embeddings([question])
        _, indices = index.search(q_embed, k=2)
        relevant = " ".join([chunks[i] for i in indices[0] if i < len(chunks)])

        context = relevant[:Config.MAX_CONTEXT_LENGTH]

        prompt = f"""
        You are an expert in insurance documents. Answer the question strictly based on the context.
        Be concise and accurate in {language}.

        Context: {context}
        Question: {question}
        Answer:
        """

        return await generate_with_gemini(prompt)
    except Exception as e:
        logging.error(f"Question processing failed: {str(e)}")
        return f"Error: Could not process question."

@app.post("/hackrx/run", response_model=DocumentResponse)
async def answer_questions(request: DocumentRequest):
    start_time = time.time()
    try:
        # Step 1: Download PDF
        pdf_bytes = await download_pdf(request.documents)
        raw_text = extract_text_from_pdf(pdf_bytes)

        # Step 2: Chunk text
        text_chunks = chunk_text(raw_text)
        if not text_chunks:
            raise HTTPException(status_code=422, detail="No text extracted from PDF")

        # Step 3: Process questions
        semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
        async def limited_question(q):
            async with semaphore:
                return await process_single_question(q, text_chunks, request.language)

        answers = await asyncio.gather(*[limited_question(q) for q in request.questions])

        total_time = time.time() - start_time
        rationale = (
            f"Processed {len(request.questions)} questions in {total_time:.2f}s. "
            f"Used Gemini 1.5 Flash. Analyzed {len(text_chunks)} chunks."
        )

        return DocumentResponse(answers=answers, rationale=rationale)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "model": "gemini-1.5-flash",
        "embedding_model": Config.EMBEDDING_MODEL,
        "memory_safe": True
    }

# Graceful shutdown
@app.on_event("shutdown")
def shutdown_event():
    global embedding_model, executor
    del embedding_model
    if executor:
        executor.cancel()

# Server entrypoint
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")