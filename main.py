from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

app = FastAPI()

# Allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.onrender.com"],  # replace with your frontend URL
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load everything once at startup
print("Loading model and index...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index/index.bin")
chunks = np.load("faiss_index/chunks.npy", allow_pickle=True)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("Ready!")

class Question(BaseModel):
    question: str

@app.post("/chat")
def chat(body: Question):
    # Embed the question
    query_embedding = model.encode([body.question])

    # Retrieve top 3 chunks
    distances, indices = index.search(np.array(query_embedding), k=3)
    context = "\n\n".join(chunks[i] for i in indices[0])

    prompt = f"""You are a helpful assistant for Notable Media's website.
Answer ONLY using the provided context.
If the answer is not in the context, say: 'I could not find that information.'

Context:
{context}

Question:
{body.question}"""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    return {"answer": response.choices[0].message.content}

@app.get("/")
def root():
    return {"status": "RAG backend running"}