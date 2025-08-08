# main.py
import os
import tempfile
import PyPDF2
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv
from urllib.request import urlopen

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable debug logs from langchain to reduce noise
logging.getLogger("langchain").setLevel(logging.ERROR)

# Initialize FastAPI
app = FastAPI(title="LLM-Powered Policy Query System", version="1.1")

# Allow CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 20,
    "max_output_tokens": 512,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Using the most efficient Gemini model for cost and performance
try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    logger.info("Successfully configured Gemini 1.5 Flash.")
except Exception as e:
    logger.error(f"Failed to configure Gemini model: {e}")
    raise HTTPException(status_code=500, detail="LLM configuration failed.")

# Use lightweight local embeddings to save cost and memory
# This model is a good balance of performance and size for free-tier services
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Cache for document index (simple in-memory cache)
document_index_cache = {}

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def run_query(request: QueryRequest):
    """
    Processes a list of questions against a document URL, returning answers.
    
    This endpoint downloads a PDF, processes its content using FAISS and
    HuggingFace embeddings, and then uses the Gemini LLM to answer
    questions based on the extracted context. The FAISS index is cached
    to improve performance on subsequent calls with the same document.
    """
    try:
        doc_url = request.documents.strip()
        questions = [q.strip() for q in request.questions if q.strip()]

        if not doc_url or not questions:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing documents or questions")

        logger.info(f"Processing {len(questions)} questions for document: {doc_url}")

        # Reuse index if already loaded, for efficiency
        if doc_url in document_index_cache:
            logger.info("Using cached document index.")
            vectorstore = document_index_cache[doc_url]
        else:
            logger.info("Document not cached. Starting download and processing.")
            
            # Use a temporary file to handle the PDF download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                try:
                    resp = urlopen(doc_url)
                    tmpfile.write(resp.read())
                    tmp_path = tmpfile.name
                except Exception as e:
                    os.unlink(tmpfile.name)
                    logger.error(f"Failed to download PDF from URL: {e}")
                    raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=f"Failed to download document: {e}")

            # Extract text from the PDF
            texts = []
            try:
                with open(tmp_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            texts.append(text)
            finally:
                os.unlink(tmp_path)  # Clean up temp file

            if not texts:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No text found in PDF")

            # Split text into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.create_documents(texts)

            # Create FAISS index and cache it
            vectorstore = FAISS.from_documents(chunks, embeddings)
            document_index_cache[doc_url] = vectorstore
            logger.info("Document processed and cached successfully.")

        # Process each question and generate an answer
        answers = []
        for question in questions:
            # Retrieve relevant context from the FAISS index
            relevant_docs = vectorstore.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Use a fine-tuned prompt to get accurate, policy-based answers
            prompt = f"""
            You are an expert policy analyst. Answer the question strictly based on the context below.
            Be concise, accurate, and quote key conditions. If unsure, say 'Not specified'.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """

            try:
                response = model.generate_content(prompt)
                answer = response.text.strip()
            except Exception as e:
                logger.error(f"LLM error for question '{question}': {e}")
                answer = "Unable to generate response due to model error."

            answers.append(answer)

        return JSONResponse(content={"answers": answers})

    except HTTPException as http_e:
        # Re-raise HTTPException to be handled by FastAPI
        raise http_e
    except Exception as e:
        logger.error(f"Request failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"status": "LLM Query System is running", "model": "gemini-1.5-flash-latest"}