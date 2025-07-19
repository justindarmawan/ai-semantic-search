from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

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

    def add_document(self, text):
        self.vector_db.add_texts([text])

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
