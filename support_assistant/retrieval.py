"""Local sentence-transformer embeddings persisted in ChromaDB."""
from pathlib import Path
import os
DOCS_DIR=Path(__file__).with_name("documents")
DB_DIR=Path(__file__).with_name("chroma_db")
COLLECTION_NAME="zepto_policy_docs"

def load_documents():
    return [{"source":p.name,"text":p.read_text(encoding="utf-8").strip()} for p in sorted(DOCS_DIR.glob("*.txt"))]

def build_or_load_vector_store():
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to enable local RAG (chromadb and sentence-transformers).") from exc
    model=SentenceTransformer(os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2"))
    client=chromadb.PersistentClient(path=str(DB_DIR))
    try: client.delete_collection(COLLECTION_NAME)
    except Exception: pass
    collection=client.create_collection(COLLECTION_NAME,metadata={"hnsw:space":"cosine"})
    docs=load_documents()
    if not docs: raise RuntimeError(f"No policy documents found in {DOCS_DIR}")
    vectors=model.encode([d["text"] for d in docs],normalize_embeddings=True).tolist()
    collection.add(ids=[d["source"] for d in docs],documents=[d["text"] for d in docs],metadatas=[{"source":d["source"]} for d in docs],embeddings=vectors)
    return collection,model

_STORE=None
def get_vector_store():
    global _STORE
    if _STORE is None: _STORE=build_or_load_vector_store()
    return _STORE

def retrieve(query,top_k=3):
    collection,model=get_vector_store()
    vector=model.encode([query],normalize_embeddings=True).tolist()[0]
    result=collection.query(query_embeddings=[vector],n_results=min(top_k,collection.count()),include=["documents","metadatas","distances"])
    items=[]
    for doc,meta,distance in zip(result["documents"][0],result["metadatas"][0],result["distances"][0]):
        items.append({"text":doc,"source":meta.get("source","unknown"),"distance":float(distance)})
    return items
