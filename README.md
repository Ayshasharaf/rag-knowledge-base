# RAG Knowledge Base API

An AI-powered REST API for uploading PDF documents, indexing them with embeddings, and answering questions using Retrieval Augmented Generation (RAG).

## How It Works

1. Upload a PDF document to `/upload`
2. The document is split into text chunks and embedded with Hugging Face embeddings
3. Chunks are stored and searched in Pinecone
4. Relevant chunks are retrieved and passed to an LLM to generate an answer

## Tech Stack

- **FastAPI** — REST API framework
- **LangChain** — Document splitting and embedding orchestration
- **Pinecone** — Vector database for similarity search
- **Hugging Face embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
- **OpenAI-compatible API** — used via Hugging Face router for chat completions
- **Docker** — Containerized deployment

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload and index a PDF file |
| POST | `/ask` | Ask a question using indexed documents |

## Usage

### Upload a PDF

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/document.pdf"
```

### Ask a question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run with Docker

```bash
docker build -t rag-knowledge-base .
docker run -p 8000:8000 --env-file .env rag-knowledge-base
```

## Environment Variables

Create a `.env` file with these values:

```bash
HF_API_KEY=your_huggingface_token
HF_TOKEN=your_huggingface_token
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=rag-knowledge-base
```

## Notes

- Only PDF uploads are supported.
- Uploaded documents are stored in `uploaded_docs/`.
- The application expects a Pinecone index configured with `PINECONE_INDEX`.
- Answers are generated using the model configured in `app/generator.py`.
