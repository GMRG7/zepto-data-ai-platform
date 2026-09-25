"""LangGraph workflow with deterministic, policy-grounded mock responses."""
from typing import TypedDict, List, Dict, Any
import os
from langgraph.graph import StateGraph, END

class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: List[str]
    confidence: float

def classify_intent(state:AssistantState)->AssistantState:
    q=state["query"].lower()
    direct_terms=("hello","hi","hey","thank you","thanks","good morning","good evening")
    intent="direct" if any(t in q.strip() for t in direct_terms) and len(q.split())<8 else "policy"
    return {"intent":intent}

def direct_answer(state:AssistantState)->AssistantState:
    q=state["query"].lower()
    answer="Hello! I can help with delivery, orders, returns, memberships, gift cards, and support policies."
    if "thank" in q: answer="You're welcome! Let me know if you have another question."
    return {"answer":answer,"sources":[],"confidence":1.0}

def retrieve_and_answer(state:AssistantState)->AssistantState:
    from retrieval import retrieve
    matches=retrieve(state["query"],top_k=3)
    if not matches: return {"answer":"I couldn't find a matching policy in the available documents. Please contact Zepto support through the app.","sources":[],"confidence":0.0}
    # Mock answer is grounded in the highest-ranked document, avoiding unsupported synthesis.
    best=matches[0]
    answer=best["text"].strip()
    sources=list(dict.fromkeys(m["source"] for m in matches[:2]))
    confidence=max(0.0,min(1.0,1.0-float(best["distance"])))
    return {"answer":answer,"sources":sources,"confidence":round(confidence,3)}

def _route(state:AssistantState): return state.get("intent","policy")
def build_graph():
    graph=StateGraph(AssistantState)
    graph.add_node("classify_intent",classify_intent)
    graph.add_node("direct_answer",direct_answer)
    graph.add_node("retrieve_and_answer",retrieve_and_answer)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent",_route,{"direct":"direct_answer","policy":"retrieve_and_answer"})
    graph.add_edge("direct_answer",END); graph.add_edge("retrieve_and_answer",END)
    return graph.compile()

_graph=None
def get_graph():
    global _graph
    if _graph is None: _graph=build_graph()
    return _graph

def run_assistant(query:str)->dict:
    result=get_graph().invoke({"query":query})
    return {"answer":result.get("answer",""),"sources":result.get("sources",[]),"confidence":float(result.get("confidence",0.0))}
