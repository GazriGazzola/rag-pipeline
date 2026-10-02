import os
import uuid
import requests
from bs4 import BeautifulSoup
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader

QDRANT_HOST = "http://localhost:6333"
OLLAMA_HOST = "http://localhost:11434"
COLLECTION_NAME = "enterprise_knowledge"
EMBED_MODEL = "nomic-embed-text"

qdrant_client = QdrantClient(url=QDRANT_HOST, check_compatibility=False)
ollama_client = ollama.Client(host=OLLAMA_HOST)

def init_vector_db():
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

def extract_text_from_pdf(filepath):
    text = ""
    try:
        reader = PdfReader(filepath)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        print(f"Failed to read PDF {filepath}: {e}")
    return text

def extract_text_from_url(url):
    try:
        # Fetch the webpage
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Rip out all the JavaScript and CSS styles
        for script in soup(["script", "style"]):
            script.extract()
            
        # Grab only the visible text
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""

def ingest_url(url):
    init_vector_db()
    print(f"Scraping {url}...")
    content = extract_text_from_url(url)
    
    if not content:
        print("No readable text found. Skipping.")
        return

    print(f"Embedding scraped content...")
    response = ollama_client.embed(model=EMBED_MODEL, input=content)
    
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=response['embeddings'][0],
        payload={"text": f"Source: {url}\n\n{content}"}
    )
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
    print(f"\nSuccessfully ingested {url} into Qdrant.")

def ingest_directory(directory_path: str):
    init_vector_db()
    files = [f for f in os.listdir(directory_path) if f.endswith(('.txt', '.md', '.py', '.csv', '.pdf'))]
    if not files:
        return

    for filename in files:
        filepath = os.path.join(directory_path, filename)
        if filename.endswith('.pdf'):
            content = extract_text_from_pdf(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        
        if not content.strip():
            continue

        response = ollama_client.embed(model=EMBED_MODEL, input=content)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=response['embeddings'][0],
            payload={"text": f"Source: {filename}\n\n{content}"}
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        
    print(f"Successfully ingested {len(files)} files into Qdrant.")

if __name__ == "__main__":
    print("--- Enterprise RAG Ingestion Engine ---")
    
    # 1. Process local files
    DOCS_DIR = "./documents"
    os.makedirs(DOCS_DIR, exist_ok=True)
    ingest_directory(DOCS_DIR)
    
    # 2. Ask if you want to scrape a website
    target_url = input("\nEnter a URL to scrape (or press Enter to skip): ")
    if target_url.strip():
        ingest_url(target_url.strip())