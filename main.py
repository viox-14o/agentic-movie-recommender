import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm import TOP_MOVIES, get_recommendation

# ---------------------------------------------------------------------------
# DO NOT EDIT: FastAPI app and request/response schemas
#
# These define the API contract. Changing them will break the grader.
# ---------------------------------------------------------------------------

app = FastAPI(title="Movie Recommender")


class WatchHistoryItem(BaseModel):
    tmdb_id: int
    name: str


class RecommendRequest(BaseModel):
    user_id: int
    preferences: str
    history: list[WatchHistoryItem] = []


class RecommendResponse(BaseModel):
    tmdb_id: int
    user_id: int
    description: str


# ---------------------------------------------------------------------------
# DO NOT EDIT: Endpoint
#
# Calls get_recommendation() from llm.py and enforces the output contract
# (valid tmdb_id, description ≤500 chars). Edit llm.py instead.
# ---------------------------------------------------------------------------


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    history_names = [h.name for h in request.history]
    try:
        result = get_recommendation(request.preferences, history_names)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    valid_ids = set(TOP_MOVIES["tmdb_id"].astype(int))
    tmdb_id = int(result.get("tmdb_id", -1))
    if tmdb_id not in valid_ids:
        raise HTTPException(
            status_code=502, detail=f"LLM returned invalid tmdb_id: {tmdb_id}"
        )

    description = str(result.get("description", ""))[:500]

    return RecommendResponse(
        tmdb_id=tmdb_id,
        user_id=request.user_id,
        description=description,
    )


@app.get("/")
def health():
    return {"status": "ok", "candidates": len(TOP_MOVIES)}
