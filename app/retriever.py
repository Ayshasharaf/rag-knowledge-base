from app.ingestor import load_vector_store

def retrieve_relevant_chunks(question: str, k: int = 3) -> list[str]:
    vector_store = load_vector_store()
    
    results = vector_store.similarity_search(question, k=k)
    
    chunks = [doc.page_content for doc in results]
    return chunks