import fitz  # PyMuPDF
import requests
import faiss
import numpy as np
from sentence_splitter import split_text_into_sentences
from utils import download_pdf, gemini_embed

def get_relevant_chunks(pdf_url, questions):
    raw_text = download_pdf(pdf_url)
    sentences = split_text_into_sentences(raw_text)

    # Embed sentences using Gemini
    sentence_vectors = [gemini_embed(s) for s in sentences]
    dim = len(sentence_vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(sentence_vectors))

    results = []
    for q in questions:
        q_vec = np.array(gemini_embed(q)).reshape(1, -1)
        D, I = index.search(q_vec, 3)  # Top 3 relevant chunks
        context = "\n".join([sentences[i] for i in I[0]])
        results.append(context)

    return results
