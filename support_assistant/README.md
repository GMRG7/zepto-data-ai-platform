# Module 3 — Support Assistant

Implement an offline-first RAG assistant over the eight supplied policy documents. Graded default is `MOCK_LLM=1` (no external LLM calls). Use local `all-MiniLM-L6-v2` embeddings and ChromaDB retrieval, LangGraph nodes and conditional routing, Pydantic-validated JSON output, FastAPI `/ask`, and a buildable Dockerfile.

Run locally after implementation:
```bash
uvicorn app:app --reload
```
