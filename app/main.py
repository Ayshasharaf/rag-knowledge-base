import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.ingestor import ingest_document
from app.retriever import retrieve_relevant_chunks
from app.generator import generate_answer

app = FastAPI(title="RAG Knowledge Base API")

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionInput(BaseModel):
    question: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    chunks_count = ingest_document(file_path)
    
    return {
        "message": "Document uploaded and indexed successfully",
        "filename": file.filename,
        "chunks_indexed": chunks_count
    }

@app.post("/ask")
def ask_question(body: QuestionInput):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    chunks = retrieve_relevant_chunks(body.question)
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context found")
    
    answer = generate_answer(body.question, chunks)
    
    return {
        "question": body.question,
        "answer": answer,
        "sources": chunks
    }