import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from rag_pipeline import explain_graph_rag

app = FastAPI(title="MediGraph RAG Explainability API")

# Setup CORS for the Vite React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since it's a local tool, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GraphPayload(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

@app.post("/api/explain-graph")
async def explain_graph(payload: GraphPayload):
    try:
        # Check if the OpenAI API Key is correctly set
        if not os.environ.get("OPENAI_API_KEY"):
            return {"rationale": "Configuration Error: OPENAI_API_KEY is not set on the server environment. Please set it before generating an AI explanation."}
            
        rationale = explain_graph_rag(payload.nodes, payload.edges)
        return {"rationale": rationale}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
