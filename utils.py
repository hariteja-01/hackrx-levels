import os
from dotenv import load_dotenv
import fitz
import requests

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def download_pdf(url):
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    doc = fitz.open("temp.pdf")
    return "\n".join([page.get_text() for page in doc])

def gemini_embed(text):
    URL = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={API_KEY}"
    payload = {"model": "models/embedding-001", "content": {"parts": [{"text": text}]}}

    res = requests.post(URL, json=payload)
    return res.json()["embedding"]["values"]
