# Local Enterprise RAG Pipeline

A fully containerized, locally hosted Retrieval-Augmented Generation (RAG) architecture designed to ingest, embed, and interrogate unstructured data (PDFs, raw text, and live web scrapes) without sending telemetry or data to external APIs. 

Built as a practical demonstration of systems engineering, backend containerization, and Python-based API integration.

## Architecture & Tech Stack

*   **Frontend Interface:** Streamlit (Python)
*   **LLM Backend:** Ollama running **Llama 3.2** (3B)
*   **Vector Database:** Qdrant (Containerized via Docker)
*   **Embedding Model:** `nomic-embed-text`
*   **Ingestion Engine:** `BeautifulSoup4` (HTML/Web scraping), `PyPDF` (Binary document parsing)
*   **Infrastructure:** Docker Compose with persistent volume mapping

## Key Features

*   **100% Local Execution:** The entire stack runs completely offline, ensuring absolute data privacy for sensitive IT documentation, CVs, or corporate logs.
*   **Hardware Optimized:** Specifically engineered to operate efficiently within an 8GB VRAM constraint (RTX 3070), balancing system memory and GPU acceleration to prevent CUDA out-of-memory errors.
*   **Multi-Format Ingestion:** Python ingestion scripts automatically strip binary formatting and CSS/JS from PDFs and live URLs, converting raw information into mathematical vector embeddings.
*   **Persistent Conversational Memory:** The Streamlit UI actively injects chat history back into the LLM payload for contextual follow-up questions, complete with a manual kill-switch to wipe memory and prevent token overflow.
*   **Dynamic Inference Control:** A real-time UI slider directly manipulates the LLM's temperature setting, allowing on-the-fly adjustments to the neural network's hallucination probability.

## Deployment Instructions

### Prerequisites
* Docker & Docker Compose
* Python 3.10+
* Minimum 8GB VRAM (NVIDIA GPU recommended for CUDA acceleration)

### 1. Spin Up the Backend
Initialize the Qdrant vector database and Ollama LLM containers. The `docker-compose.yml` ensures persistent volume storage so embedded data survives reboots.
```bash
docker compose up -d
2. Install Python Dependencies
Bash
pip install streamlit ollama qdrant-client pypdf requests beautifulsoup4
3. Ingest Data
Place any .pdf, .txt, or .csv files into the documents/ directory, or use the interactive prompt to scrape a live URL.

Bash
python ingest.py
4. Launch the Frontend
Bypass local application restrictions by serving the UI directly through a localized web port.

Bash
python -m streamlit run app.py
Security Note
This repository includes a strict .gitignore to prevent the accidental upload of personal data, API keys, or locally mapped Docker volumes. The documents/ directory is deliberately untracked.
