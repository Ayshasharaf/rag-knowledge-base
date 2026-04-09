import os
from app.ingestor import get_embeddings, get_pinecone_index

def retrieve_relevant_chunks(question: str, k: int = 3) -> list[str]:
    embeddings = get_embeddings()
    index = get_pinecone_index()

    question_embedding = embeddings.embed_query(question)

    results = index.query(
        vector=question_embedding,
        top_k=k,
        include_metadata=True
    )

    chunks = [match["metadata"]["text"] for match in results["matches"]]
    return chunks