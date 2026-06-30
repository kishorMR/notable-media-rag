from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
import cohere
from groq import Groq
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://notable-media-frontend.onrender.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading index...")
index = faiss.read_index("faiss_index/index.bin")
chunks = np.load("faiss_index/chunks.npy", allow_pickle=True)
co = cohere.Client(os.environ.get("COHERE_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("Ready!")

class Question(BaseModel):
    question: str

@app.post("/chat")
def chat(body: Question):
    response = co.embed(
        texts=[body.question],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    query_embedding = np.array(response.embeddings, dtype="float32")

    distances, indices = index.search(query_embedding, k=3)
    context = "\n\n".join(chunks[i] for i in indices[0])

    prompt = f"""You are a helpful assistant for Notable Media's website.
Answer ONLY using the provided context.
If the answer is not in the context, say: 'I could not find that information.'

Context:
{context}

Question:
{body.question}"""

    completion = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    return {"answer": completion.choices[0].message.content}

@app.get("/")
def root():
    return {"status": "RAG backend running"}