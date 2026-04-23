from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from llm import get_recommendation

app = FastAPI()


class HistoryItem(BaseModel):
    tmdb_id: int
    name: str


class RecommendationRequest(BaseModel):
    user_id: int
    preferences: str
    history: List[HistoryItem] = []


@app.post("/")
async def recommend(request: RecommendationRequest):
    history_names = [item.name for item in request.history]
    history_ids = [item.tmdb_id for item in request.history]

    try:
        result = get_recommendation(request.preferences, history_names, history_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "tmdb_id": int(result["tmdb_id"]),
        "user_id": request.user_id,
        "description": str(result.get("description", ""))[:500],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
