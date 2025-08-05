import os
import re
import time
import logging
import fitz  # PyMuPDF
import faiss
import numpy as np
import hashlib
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import asyncio
import aiohttp
from dotenv import load_dotenv
import backoff
import google.generativeai as genai
from async_lru import alru_cache

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- OPTIMIZED: Configuration ---
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = "models/embedding-001"
    GENERATION_MODEL = "gemini-1.5-flash"
    # Tuned for better context without losing meaning
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_CONTEXT_LENGTH = 12000 # Increased context for better answers
    PDF_DOWNLOAD_TIMEOUT = 20
    MAX_RETRIES = 3
    EMBEDDING_DIMENSION = 768
    # Number of chunks to retrieve for context
    RETRIEVAL_TOP_K = 7
    # Batch size for embedding API calls
    EMBEDDING_BATCH_SIZE = 100

# Validate config and configure Gemini SDK
if not Config.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=Config.GEMINI_API_KEY)

# FAISS index (re-created for each request)
faiss_index = None

# FastAPI app
app = FastAPI(
    title="High-Performance Document Q&A with Gemini",
    description="Optimized RAG pipeline for maximum accuracy and speed.",
    version="3.0.0"
)

# Data models
class DocumentRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    language: str = "en"

class DocumentResponse(BaseModel):
    answers: List[str]
    rationale: str

# --- UPGRADED: Utility functions ---
def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)-\s*(\w)', r'\1\2', text)
    return text.strip()

# --- UPGRADED: Sentence-aware chunking for higher accuracy ---
def chunk_text(text: str) -> List[str]:
    # Split by paragraphs first, then sentences
    single_chunks = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    for p in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', p)
        single_chunks.extend([s for s in sentences if s])

    # Combine sentences into semantic chunks
    combined_chunks = []
    current_chunk = ""
    for sentence in single_chunks:
        if len(current_chunk) + len(sentence) + 1 < Config.CHUNK_SIZE:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                combined_chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        combined_chunks.append(current_chunk.strip())
        
    logger.info(f"Created {len(combined_chunks)} semantic chunks from text.")
    return combined_chunks

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
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return preprocess_text(" ".join(page.get_text() for page in doc))
    except Exception as e:
        logger.error(f"PDF text extraction failed: {str(e)}")
        raise HTTPException(status_code=422, detail="Failed to extract text from PDF.")

# --- UPGRADED: Batch embedding with caching ---
@alru_cache(maxsize=128)
@backoff.on_exception(backoff.expo, Exception, max_tries=Config.MAX_RETRIES)
async def get_batch_embeddings(texts: List[str], task_type: str) -> List[np.ndarray]:
    try:
        # Gemini API call for batch embeddings
        result = await genai.embed_content_async(
            model=Config.EMBEDDING_MODEL,
            content=texts,
            task_type=task_type
        )
        return [np.array(e) for e in result['embedding']]
    except Exception as e:
        logger.error(f"Gemini batch embedding failed: {str(e)}")
        raise

async def build_faiss_index(text_chunks: List[str]):
    global faiss_index
    try:
        # Process in batches to handle large documents
        all_embeddings = []
        for i in range(0, len(text_chunks), Config.EMBEDDING_BATCH_SIZE):
            batch_texts = text_chunks[i:i + Config.EMBEDDING_BATCH_SIZE]
            embeddings_batch = await get_batch_embeddings(tuple(batch_texts), "RETRIEVAL_DOCUMENT")
            all_embeddings.extend(embeddings_batch)
        
        embeddings_np = np.array(all_embeddings).astype('float32')
        faiss_index = faiss.IndexFlatL2(Config.EMBEDDING_DIMENSION)
        faiss_index.add(embeddings_np)
        logger.info(f"FAISS index built successfully with {faiss_index.ntotal} vectors.")
    except Exception as e:
        logger.error(f"FAISS index build failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to build document index.")

async def retrieve_relevant_chunks(question: str, text_chunks: List[str]) -> List[str]:
    try:
        # Use batch embedder for a single question (benefits from caching)
        question_embedding = (await get_batch_embeddings(tuple([question]), "RETRIEVAL_QUERY"))[0]
        question_embedding_np = question_embedding.reshape(1, -1).astype('float32')
        
        _, indices = faiss_index.search(question_embedding_np, Config.RETRIEVAL_TOP_K)
        
        return [text_chunks[idx] for idx in indices[0] if 0 <= idx < len(text_chunks)]
    except Exception as e:
        logger.error(f"Chunk retrieval failed: {e}")
        return []

# --- UPGRADED: Advanced prompt for higher accuracy ---
async def generate_answer(question: str, context: str, language: str) -> str:
    prompt = f"""You are a world-class insurance document analysis AI. Your task is to provide a precise and factual answer to the question based *exclusively* on the provided context.

Follow these instructions carefully:
1.  **Analyze the Context**: Read the entire context provided below to understand the information it contains.
2.  **Identify Key Information**: Find the specific sentences or phrases in the context that directly answer the question.
3.  **Formulate the Answer**: Construct a concise answer in {language}.
4.  **Strict Grounding**: Do NOT use any information outside of the provided context. If the answer cannot be found in the context, you MUST respond with "The answer to this question is not found in the provided document."

**Context:**
---
{context}
---

**Question:** {question}

**Answer:**
"""
    try:
        model = genai.GenerativeModel(Config.GENERATION_MODEL)
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        logger.warning(f"Answer generation failed: {str(e)}")
        return f"Could not generate an answer due to an API error: {str(e)}"

# --- UPGRADED: Caching at the question-processing level ---
@alru_cache(maxsize=256)
async def process_single_question(doc_id: str, question: str, text_chunks: List[str], language: str) -> str:
    # `doc_id` is used by the cache to differentiate questions about different documents
    try:
        relevant_chunks = await retrieve_relevant_chunks(question, text_chunks)
        if not relevant_chunks:
            return "The answer to this question is not found in the provided document."
            
        context = "\n\n".join(relevant_chunks)[:Config.MAX_CONTEXT_LENGTH]
        return await generate_answer(question, context, language)
    except Exception as e:
        logger.error(f"Question processing failed for '{question}': {str(e)}")
        return "An error occurred while processing the question."

# --- Main API Endpoint ---
@app.post("/hackrx/run", response_model=DocumentResponse)
async def answer_questions(request: DocumentRequest):
    try:
        start_time = time.time()
        
        pdf_bytes = await download_pdf(request.documents)
        # Create a unique ID for the document content for caching
        doc_id = hashlib.sha256(pdf_bytes).hexdigest()

        pdf_text = extract_text_from_pdf(pdf_bytes)
        text_chunks = chunk_text(pdf_text)
        
        if not text_chunks:
            raise HTTPException(status_code=422, detail="Document is empty or unreadable.")

        await build_faiss_index(text_chunks)
        
        # Process questions concurrently
        tasks = [
            process_single_question(doc_id, q, tuple(text_chunks), request.language)
            for q in request.questions
        ]
        answers = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        rationale = (
            f"Successfully processed {len(request.questions)} questions in {total_time:.2f}s. "
            f"Model: {Config.GENERATION_MODEL}. "
            f"Analyzed {len(text_chunks)} semantic chunks. "
            f"Performance optimizations: Batch Embeddings, Semantic Chunking, Advanced Prompting, Caching."
        )
        
        return DocumentResponse(answers=list(answers), rationale=rationale)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error in /hackrx/run: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")

@app.get("/health")
async def health_check():
    return {"status": "operational", "model": Config.GENERATION_MODEL, "optimizations_enabled": True}

if __name__ == "__main__":
    import uvicorn
    # Note: for production, consider a more robust server like uvicorn with gunicorn workers
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")