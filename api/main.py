from fastapi import FastAPI, UploadFile, Form
from services.rag_pipeline import RAGPipeline

app = FastAPI(title="AI Semantic Search API")
rag = RAGPipeline()

@app.post("/documents")
async def upload_document(file: UploadFile):
    content = await file.read()
    rag.add_document(content.decode("utf-8"))
    return {"status": "document indexed"}

@app.post("/query")
async def query_semantic_search(question: str = Form(...)):
    answer = rag.query(question)
    return {"answer": answer}

@app.get("/health")
async def health():
    return {"status": "ok"}
