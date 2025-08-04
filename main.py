from fastapi import FastAPI, Request, Header
from pydantic import BaseModel
from embedding import get_relevant_chunks
from prompt_engine import query_llm

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def run_submission(request: Request, payload: QueryRequest, authorization: str = Header(None)):
    assert authorization.startswith("Bearer "), "Missing Bearer token"
    
    try:
        # 🔹 Step 1: Embed and retrieve chunks
        chunks = get_relevant_chunks(payload.documents, payload.questions)

        # 🔹 Step 2: Query LLM with context and questions
        answers = query_llm(chunks, payload.questions)

        return {"answers": answers}
    except Exception as e:
        return {"error": str(e)}
