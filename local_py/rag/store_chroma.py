import os
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

RAG_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(RAG_DIR, "chroma_db")
COLLECTION_NAME = "local_docs"

# We must accept the global model to perform embeddings
class ChromaStore:
    def __init__(self, model):
        self.model = model
        self.client = chromadb.PersistentClient(path=DB_PATH)
        
        class LocalEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                return model.encode(input).tolist()
            def name(self) -> str:
                return "LocalMiniLM"
                
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, 
            embedding_function=LocalEmbeddingFunction()
        )

    def upsert_documents(self, ids, contents, metadatas, embeddings=None):
        # We don't strictly need to pass embeddings because the EmbeddingFunction handles it,
        # but to keep the interface consistent with PG, we can just use the built-in mechanism.
        self.collection.upsert(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )

    def search(self, query_text, n_results=4):
        # Chroma takes care of embedding the query text internally
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                formatted_results.append(f"來自文檔 [{meta['source']}]:\n{doc}")
        
        return "\n\n-----分割線-----\n\n".join(formatted_results) if formatted_results else "找不到與此問題相關的本地文檔內容。"
