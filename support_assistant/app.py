"""FastAPI application exposing health and policy question endpoints."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from graph import run_assistant
app=FastAPI(title="Zepto Support Assistant",version="1.0.0")
class AskRequest(BaseModel):
    query:str=Field(min_length=1,max_length=2000)
class AskResponse(BaseModel):
    answer:str
    sources:list[str]
    confidence:float=Field(ge=0,le=1)
@app.get("/health")
def health(): return {"status":"ok"}
@app.post("/ask",response_model=AskResponse)
def ask(payload:AskRequest):
    try: return AskResponse(**run_assistant(payload.query.strip()))
    except Exception as exc: raise HTTPException(status_code=503,detail=f"Assistant unavailable: {exc}") from exc
