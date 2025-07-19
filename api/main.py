from fastapi import FastAPI, UploadFile, Form, HTTPException
from services.rag_pipeline import RAGPipeline

app = FastAPI(title="AI Semantic Search API")
rag = RAGPipeline()

@app.post("/documents")
async def upload_document(file: UploadFile):
    try:
        content = await file.read()
        rag.add_document(content, file.filename)
        return {"status": "document indexed"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_semantic_search(question: str = Form(...)):
    try:
        answer = rag.query(question)
        return {"answer": answer}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
