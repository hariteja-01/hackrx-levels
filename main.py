import os
import re
import time
import logging
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import asyncio
import aiohttp
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import backoff
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- MODIFIED: Configuration ---
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # REMOVED: Local embedding model
    # ADDED: Gemini embedding and generation model names
    EMBEDDING_MODEL = "models/embedding-001"
    GENERATION_MODEL = "gemini-1.5-flash" # Updated to a modern, efficient model
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CONTEXT_LENGTH = 10000
    PDF_DOWNLOAD_TIMEOUT = 20 # Increased timeout for larger files
    MAX_RETRIES = 3
    # ADDED: Embedding dimension for Google's model
    EMBEDDING_DIMENSION = 768

# Validate config and configure Gemini SDK
if not Config.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=Config.GEMINI_API_KEY)

# --- REMOVED: Loading SentenceTransformer model ---
# This was the source of the memory issue.

# FAISS index (initialized globally, built per request)
faiss_index = faiss.IndexFlatL2(Config.EMBEDDING_DIMENSION)

# FastAPI app
app = FastAPI(
    title="Document Q&A with Gemini 1.5 Flash",
    description="AI-powered insurance document analysis with RAG",
    version="2.1.0"
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
            detail=f"Failed to download or validate PDF from URL: {str(e)}"
        )

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return preprocess_text("".join(page.get_text() for page in doc))
    except Exception as e:
        logger.error(f"PDF text extraction failed: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail="Failed to extract text from PDF document."
        )

# --- MODIFIED: Embedding function uses Gemini API ---
@backoff.on_exception(backoff.expo, Exception, max_tries=Config.MAX_RETRIES)
async def get_embeddings(text: str, task_type: str) -> np.ndarray:
    try:
        # Note: Using 'task_type' for retrieval is best practice
        result = await genai.embed_content_async(
            model=Config.EMBEDDING_MODEL,
            content=text,
            task_type=task_type # "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
        )
        return np.array(result['embedding'])
    except Exception as e:
        logger.error(f"Gemini embedding API request failed: {str(e)}")
        raise

# --- MODIFIED: FAISS index building is now async ---
async def build_faiss_index(text_chunks: List[str]):
    try:
        # Generate embeddings for all document chunks concurrently
        embedding_coroutines = [get_embeddings(chunk, "RETRIEVAL_DOCUMENT") for chunk in text_chunks]
        embeddings = await asyncio.gather(*embedding_coroutines)
        
        embeddings_np = np.array(embeddings).astype('float32')

        global faiss_index
        faiss_index = faiss.IndexFlatL2(Config.EMBEDDING_DIMENSION)
        faiss_index.add(embeddings_np)
        logger.info(f"FAISS index built successfully with {len(text_chunks)} chunks.")
    except Exception as e:
        logger.error(f"FAISS index build failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to build document index."
        )

# --- MODIFIED: Retrieval function is now async ---
async def retrieve_relevant_chunks(question: str, text_chunks: List[str], k: int = 5) -> List[str]:
    try:
        question_embedding = await get_embeddings(question, "RETRIEVAL_QUERY")
        question_embedding_np = question_embedding.reshape(1, -1).astype('float32')
        
        _, indices = faiss_index.search(question_embedding_np, k)
        
        return [text_chunks[idx] for idx in indices[0] if 0 <= idx < len(text_chunks)]
    except Exception as e:
        logger.error(f"Chunk retrieval failed: {e}")
        return []

# --- MODIFIED: Generation function uses Gemini SDK ---
@backoff.on_exception(backoff.expo, Exception, max_tries=Config.MAX_RETRIES)
async def generate_with_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(Config.GENERATION_MODEL)
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini generation API request failed: {str(e)}")
        raise

async def generate_answer(question: str, context: str, language: str) -> str:
    prompt = f"""You are an expert insurance document analyst. Your task is to answer the user's question based *only* on the provided context. Do not use any external knowledge. If the answer is not in the context, state that clearly.
    
    **Context:**
    ---
    {context}
    ---
    
    **Question:** {question}
    
    **Answer concisely in {language}:**"""
    
    try:
        return await generate_with_gemini(prompt)
    except Exception as e:
        logger.warning(f"Answer generation failed: {str(e)}")
        return f"Could not generate an answer due to an API error: {str(e)}"

# --- MODIFIED: Question processing is now fully async ---
async def process_single_question(question: str, text_chunks: List[str], language: str) -> str:
    try:
        relevant_chunks = await retrieve_relevant_chunks(question, text_chunks)
        if not relevant_chunks:
            return "Could not find relevant information in the document to answer the question."
            
        context = "\n\n".join(relevant_chunks)[:Config.MAX_CONTEXT_LENGTH]
        return await generate_answer(question, context, language)
    except Exception as e:
        logger.error(f"Question processing failed for '{question}': {str(e)}")
        return "An error occurred while processing the question."

# API endpoints
@app.post(
    "/hackrx/run",
    response_model=DocumentResponse,
    responses={
        400: {"description": "Bad Request: Invalid URL or file"},
        422: {"description": "Unprocessable Entity: Failed to parse PDF"},
        500: {"description": "Internal Server Error"}
    }
)
async def answer_questions(request: DocumentRequest):
    try:
        start_time = time.time()
        
        pdf_bytes = await download_pdf(request.documents)
        pdf_text = extract_text_from_pdf(pdf_bytes)
        text_chunks = chunk_text(pdf_text)
        
        if not text_chunks:
            raise HTTPException(status_code=422, detail="Document appears to be empty or could not be read.")

        # Build FAISS index for this request
        await build_faiss_index(text_chunks)
        
        # Process all questions concurrently
        answers = await asyncio.gather(*[
            process_single_question(q, text_chunks, request.language)
            for q in request.questions
        ])
        
        total_time = time.time() - start_time
        rationale = (
            f"Processed {len(request.questions)} questions in {total_time:.2f}s. "
            f"Used {Config.GENERATION_MODEL} and {Config.EMBEDDING_MODEL}. "
            f"Analyzed {len(text_chunks)} document chunks."
        )
        
        return DocumentResponse(answers=answers, rationale=rationale)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in /hackrx/run: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal processing error occurred: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "generation_model": Config.GENERATION_MODEL,
        "embedding_model": Config.EMBEDDING_MODEL,
        "rag_enabled": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")