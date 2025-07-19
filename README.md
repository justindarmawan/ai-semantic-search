# AI Semantic Search with FastAPI, Qdrant, and Phi-2

A **Retrieval-Augmented Generation (RAG)** pipeline for semantic search.  
Built with **FastAPI** for the API layer, **Qdrant** as the vector database, and **Ollama** (using the **Phi-2** model) for inference.  
Fully containerized with **Docker Compose**.

## Features

- Upload `.txt` and `.pdf` documents
- Index documents into Qdrant
- Perform **semantic queries** via a RAG pipeline
- Run a **local LLM** using Ollama (CPU or GPU)
- **Containerized environment** for development and deployment

## Requirements

- **Docker** & **Docker Compose**
- **NVIDIA GPU (optional)** for faster inference  
  For CPU-only, remove `runtime: nvidia` from the Ollama service in `docker-compose.yml`.

## Setup

Clone the repository:

```bash
git clone <repo-url>
cd ai-semantic-search
```

Build and start all containers:

```bash
docker compose up -d --build
```

**Running services:**

- **API**: http://localhost:8000
- **Qdrant**: http://localhost:6333
- **Ollama**: http://localhost:11434

## Development Workflow

- **`./api` folder is mounted to the container** → any code changes trigger FastAPI auto-reload.
- Adding new dependencies requires rebuilding the API image:

```bash
docker compose up -d --build api
```

## API Endpoints

### Upload a document

```bash
curl -X POST "http://localhost:8000/documents"      -F "file=@sample.txt"
curl -X POST "http://localhost:8000/documents"      -F "file=@sample2.pdf"
```

### Query the knowledge base

```bash
curl -X POST "http://localhost:8000/query"      -F "question=Who is Justin Darmawan?"
```

## Container Management

- **Stop containers**:
  ```bash
  docker compose stop
  ```
- **Start again**:
  ```bash
  docker compose start
  ```
- **Restart all**:
  ```bash
  docker compose restart
  ```
- **Remove containers & networks (keep data)**:
  ```bash
  docker compose down
  ```
- **Full reset (remove all volumes and data)**:
  ```bash
  docker compose down -v
  ```

## Notes

- **Persistent volumes `qdrant_data` and `ollama_data`** keep indexed data and model weights even if containers are stopped.
- For **CPU-only** mode, remove `runtime: nvidia` in the `ollama` service.
