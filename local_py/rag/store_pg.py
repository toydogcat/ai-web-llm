import os
from sqlalchemy import create_engine, Column, String, Integer, Text, select
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ── Database Model ────────────────────────────────────────────────────────────
class RagDocument(Base):
    __tablename__ = "local_agentic_rag_docs"
    
    id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    embedding = Column(Vector(384))

# Create table if it doesn't exist
Base.metadata.create_all(engine)

# ── Functions ─────────────────────────────────────────────────────────────────

def upsert_documents(ids, contents, metadatas, embeddings):
    """Insert or update documents in PostgreSQL."""
    session = SessionLocal()
    try:
        for doc_id, content, meta, emb in zip(ids, contents, metadatas, embeddings):
            # Check if exists
            existing = session.get(RagDocument, doc_id)
            if existing:
                existing.content = content
                existing.source = meta["source"]
                existing.embedding = emb
            else:
                new_doc = RagDocument(
                    id=doc_id,
                    source=meta["source"],
                    content=content,
                    embedding=emb
                )
                session.add(new_doc)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def search(query_embedding, n_results=4):
    """Semantic search using pgvector (<-> operator is L2 distance)."""
    session = SessionLocal()
    try:
        # Retrieve top n_results ordered by L2 distance
        stmt = select(RagDocument).order_by(RagDocument.embedding.l2_distance(query_embedding)).limit(n_results)
        results = session.execute(stmt).scalars().all()
        
        formatted_results = []
        for doc in results:
            formatted_results.append(f"來自文檔 [{doc.source}]:\n{doc.content}")
            
        return "\n\n-----分割線-----\n\n".join(formatted_results) if formatted_results else "找不到與此問題相關的本地文檔內容。"
    finally:
        session.close()
