import os
import fitz  # PyMuPDF
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import tempfile

# Load API key from environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]
    language: str

# Function to download and read PDF text
def extract_text_from_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(response.content)
            tmp_pdf_path = tmp_pdf.name

        doc = fitz.open(tmp_pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# Function to query Gemini with a prompt
def get_answers_from_gemini(doc_text, questions):
    answers = []
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    
    for question in questions:
        prompt = f"""
You are an expert in insurance documents. Based on the content below, answer the question clearly and precisely.

Document Content:
\"\"\"
{doc_text[:20000]}  # truncate if very long
\"\"\"

Question:
{question}

Answer:"""
        try:
            response = model.generate_content(prompt)
            answer_text = response.text.strip()
            answers.append(answer_text)
        except Exception as e:
            answers.append(f"Error generating answer: {e}")
    
    return answers

# Main API endpoint
@app.post("/hackrx/run")
def run_query(request: QueryRequest):
    document_text = extract_text_from_pdf(request.documents)

    if not document_text:
        return {"answers": ["Failed to retrieve document text."] * len(request.questions)}

    answers = get_answers_from_gemini(document_text, request.questions)

    return {"answers": answers}
