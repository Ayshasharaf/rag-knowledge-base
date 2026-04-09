# RAG Knowledge Base API

An AI-powered REST API that lets you upload documents and ask questions about them using Retrieval Augmented Generation (RAG).

## How It Works

1. Upload a PDF document
2. The system chunks and embeds it into a vector database (FAISS)
3. Ask a question in natural language
4. The system finds relevant chunks and generates an accurate answer

## Tech Stack

- **FastAPI** — REST API framework
- **LangChain** — RAG pipeline orchestration
- **FAISS** — Local vector database for similarity search
- **Sentence Transformers** — Document and query embeddings
- **Flan-T5** — Open source LLM for answer generation
- **Docker** — Containerized deployment

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Ask a question about uploaded docs |

## Run Locally

### Without Docker
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### With Docker
```bash
docker build -t rag-knowledge-base .
docker run -p 8000:8000 --env-file .env rag-knowledge-base
```

## Environment Variables
