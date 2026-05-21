import os
import sys
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

RAG_DIR = os.path.dirname(__file__)
# Ensure the current directory is in sys.path so we can import sibling modules
if RAG_DIR not in sys.path:
    sys.path.append(RAG_DIR)

VECTOR_DB_TYPE = os.environ.get("VECTOR_DB_TYPE", "chroma").lower()

# ── Embedding Model (Local) ──────────────────────────────────────────────────
# Loading model globally to avoid repeated loads
model = SentenceTransformer('all-MiniLM-L6-v2')

# ── Select Backend ────────────────────────────────────────────────────────────

if VECTOR_DB_TYPE == "pg":
    print("Using PostgreSQL + pgvector as Vector DB")
    import store_pg as pg_store
else:
    print("Using ChromaDB as Vector DB")
    from store_chroma import ChromaStore
    chroma_store = ChromaStore(model)

# ── Shared Document Logic ─────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def ingest_documents():
    """Scan local_py/rag for PDF/MD/TXT and add to vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    count = 0
    # Search in RAG_DIR (local_py/rag)
    for filename in os.listdir(RAG_DIR):
        file_path = os.path.join(RAG_DIR, filename)
        content = ""
        
        # Avoid processing directories or the DB itself
        if os.path.isdir(file_path):
            continue
            
        if filename.endswith(".pdf"):
            try:
                content = extract_text_from_pdf(file_path)
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")
                continue
        elif filename.endswith((".txt", ".md")):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        
        if content:
            chunks = text_splitter.split_text(content)
            ids = [f"{filename}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename} for _ in range(len(chunks))]
            
            if VECTOR_DB_TYPE == "pg":
                # Compute embeddings manually for PG
                embeddings = model.encode(chunks).tolist()
                pg_store.upsert_documents(ids, chunks, metadatas, embeddings)
            else:
                # Chroma handles embeddings internally
                chroma_store.upsert_documents(ids, chunks, metadatas)
                
            count += 1
            print(f"Indexed: {filename} ({len(chunks)} chunks)")
    
    return f"Successfully indexed {count} files."

def search_docs(query: str, n_results: int = 4):
    """Perform semantic search for a query."""
    if VECTOR_DB_TYPE == "pg":
        # For PG, we embed the query here
        query_embedding = model.encode(query).tolist()
        return pg_store.search(query_embedding, n_results)
    else:
        # For Chroma, we pass the raw text
        return chroma_store.search(query, n_results)

if __name__ == "__main__":
    # Test run
    print("--- 正在掃描並索引本地文檔 ---")
    print(ingest_documents())
    print("\n--- 測試檢索功能 ---")
    print(search_docs("請根據文檔總結核心內容"))
