import os
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import tempfile

class RAGPipeline:
    def __init__(self):
        self.collection_name = "docs"
        self.client = QdrantClient(host="qdrant", port=6333)

        if self.collection_name not in [col.name for col in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=384,  
                    distance=qdrant_models.Distance.COSINE
                )
            )

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        self.llm = Ollama(model="phi", base_url="http://ollama:11434")

    def load_document(self, file: bytes, filename: str):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".txt", ".pdf"]:
            raise ValueError("Unsupported file type. Only .txt and .pdf are allowed.")

        if ext == ".txt":
            return [file.decode("utf-8")]
        elif ext == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file)
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            os.remove(tmp_path)
            return [page.page_content for page in pages]

    def add_document(self, file: bytes, filename: str):
        texts = self.load_document(file, filename)
        self.vector_db.add_texts(texts)

    def query(self, question):
        docs = self.vector_db.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an AI assistant. 
        Answer the question strictly based on the context below. 
        If the answer is not found in the context, respond exactly with: "Not found in the document."

        CONTEXT:
        {context}

        QUESTION:
        {question}

        Provide a clear and concise answer only using the given context.
        """
        
        return self.llm(prompt)
