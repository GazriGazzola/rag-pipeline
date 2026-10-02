import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_HOST = "http://localhost:6333"
OLLAMA_HOST = "http://localhost:11434"
COLLECTION_NAME = "enterprise_knowledge"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

qdrant_client = QdrantClient(url=QDRANT_HOST, check_compatibility=False)
ollama_client = ollama.Client(host=OLLAMA_HOST)

def init_vector_db():
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

def ingest_documents(docs: list[str]):
    init_vector_db()
    for idx, doc_text in enumerate(docs):
        response = ollama_client.embed(model=EMBED_MODEL, input=doc_text)
        vector = response['embeddings'][0]
        
        point = PointStruct(
            id=idx,
            vector=vector,
            payload={"text": doc_text}
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])

def query_rag(user_question: str) -> str:
    query_vector = ollama_client.embed(model=EMBED_MODEL, input=user_question)['embeddings'][0]
    
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=2
    )
    
    retrieved_contexts = [point.payload["text"] for point in search_results.points]
    context_block = "\n---\n".join(retrieved_contexts)
    
    system_prompt = (
        "You are a helpful assistant. Use ONLY the provided Context below to answer the Question.\n"
        "If the context does not contain the answer, say 'I cannot find that in the local database.'\n\n"
        f"Context:\n{context_block}"
    )

    response = ollama_client.chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_question}
        ]
    )
    return response['message']['content']

if __name__ == "__main__":
    internal_docs = [
        "The database backup runs at 02:00 AM every night and goes to the AWS S3 Glacier storage bucket.",
        "The company holiday party is scheduled for December 15th at the main office.",
        "All IT support requests must go through the Jira Service Desk portal, not email."
    ]
    
    print("Starting Ingestion...")
    ingest_documents(internal_docs)
    
    print("\nExecuting RAG Query...")
    question = "When does the database backup run and where does it go?"
    answer = query_rag(question)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}\n")
    
    unrelated_question = "What is the capital of France?"
    print(f"Question: {unrelated_question}")
    print(f"Answer: {query_rag(unrelated_question)}\n")