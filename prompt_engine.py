import os
from dotenv import load_dotenv
import requests

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def query_llm(contexts, questions):
    MODEL = "models/gemini-pro"
    URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent?key={API_KEY}"

    answers = []

    for question, ctx in zip(questions, contexts):
        payload = {
            "contents": [{
                "parts": [
                    {"text": f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer precisely with facts from the context."}
                ]
            }]
        }

        res = requests.post(URL, json=payload)
        ans = res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        answers.append(ans.strip())

    return answers
