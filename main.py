from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai

# Read Gemini API Key from environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()

# Allow all CORS origins (for frontend testing and public access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    documents: str
    questions: list[str]
    language: str

@app.post("/hackrx/run")
async def run_query(payload: QueryRequest):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    prompt = f"""
    You are a highly intelligent assistant. You are given a policy document at this link: {payload.documents}.
    Answer the following questions based only on the information in that document.
    Questions:
    """
    for i, question in enumerate(payload.questions, 1):
        prompt += f"\n{i}. {question}"

    prompt += f"\n\nLanguage to answer in: {payload.language}"

    try:
        response = model.generate_content(prompt)
        return {
            "answers": response.text.strip()
        }
    except Exception as e:
        return {
            "error": str(e)
        }

@app.get("/")
def read_root():
    return {"status": "FastAPI HackRx backend running"}
