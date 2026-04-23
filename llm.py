"""
Movie recommendation agent using LLM with smart candidate selection.

Two-stage approach:
  1. Score all 1000 movies by genre/keyword relevance to user preferences
  2. Pass the top ~15 candidates to the LLM for final selection + description
"""

import json
import os
import re
import time
import argparse

import ollama
import pandas as pd

MODEL = "gemma4:31b-cloud"

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
# Load ALL movies so test.py's VALID_IDS covers the full 1000-movie database
TOP_MOVIES = pd.read_csv(DATA_PATH)

# Genre/theme triggers used to match user preferences against movie metadata
_GENRE_TRIGGERS = {
    "Action":          ["action", "fight", "war", "combat", "battle", "explosive", "guns"],
    "Comedy":          ["comedy", "funny", "humor", "laugh", "hilarious", "comedic", "fun"],
    "Drama":           ["drama", "emotional", "serious", "moving", "touching", "oscar"],
    "Horror":          ["horror", "scary", "terrifying", "fear", "ghost", "monster", "zombie"],
    "Romance":         ["romance", "romantic", "love story", "relationship", "dating"],
    "Science Fiction": ["science fiction", "sci-fi", "scifi", "space", "future", "robot", "alien"],
    "Animation":       ["animation", "animated", "cartoon", "pixar", "disney"],
    "Crime":           ["crime", "mystery", "detective", "noir", "heist", "murder", "whodunit"],
    "Adventure":       ["adventure", "quest", "journey", "expedition", "explore"],
    "Thriller":        ["thriller", "suspense", "tense", "psychological"],
    "Fantasy":         ["fantasy", "magic", "wizard", "witch", "dragon", "mythology"],
    "Family":          ["family", "kids", "children", "wholesome", "feel-good"],
    "Biography":       ["biography", "biopic", "true story", "based on real"],
    "Superhero":       ["superhero", "marvel", "avengers", "batman", "spider-man", "superpower"],
    "Buddy":           ["buddy cop", "buddy", "duo", "partners", "partner"],
    "Western":         ["western", "cowboy", "wild west"],
}

_STOP_WORDS = {
    "love", "like", "want", "good", "great", "movie", "film", "watch", "with",
    "that", "this", "from", "have", "been", "will", "they", "their", "there",
    "some", "also", "more", "very", "much", "most", "feel", "something", "anything",
    "really", "enjoy", "kind", "looking", "would", "could", "enjoy", "need",
}


def _select_candidates(preferences: str, history_ids: set, n: int = 15) -> pd.DataFrame:
    """Return up to N movies ranked by relevance, excluding already-seen movies.

    Fully vectorised — no Python-level row loop — so it runs in ~10ms on 1000 rows.
    """
    pref_lower = preferences.lower()

    # Build a single searchable text column per movie (vectorised string ops)
    movie_text = (
        TOP_MOVIES["genres"].fillna("").str.lower() + " "
        + TOP_MOVIES["keywords"].fillna("").str.lower() + " "
        + TOP_MOVIES["overview"].fillna("").str.lower() + " "
        + TOP_MOVIES["top_cast"].fillna("").str.lower()
    )

    score = pd.Series(0.0, index=TOP_MOVIES.index)

    # Genre/theme alignment (4 pts per matching genre)
    for genre, triggers in _GENRE_TRIGGERS.items():
        in_pref = any(t in pref_lower for t in triggers)
        if not in_pref:
            continue
        genre_lower = genre.lower()
        pattern = "|".join(re.escape(t) for t in [genre_lower] + triggers[:3])
        in_movie = movie_text.str.contains(pattern, regex=True, na=False)
        score += in_movie.astype(float) * 4.0

    # Token-level preference matching (0.6 pts per matching word)
    pref_tokens = [w for w in re.findall(r"\b[a-z]{4,}\b", pref_lower) if w not in _STOP_WORDS]
    for token in pref_tokens:
        score += movie_text.str.contains(re.escape(token), regex=True, na=False).astype(float) * 0.6

    # Quality signals
    va = pd.to_numeric(TOP_MOVIES["vote_average"], errors="coerce").fillna(0)
    vc = pd.to_numeric(TOP_MOVIES["vote_count"], errors="coerce").fillna(0)
    score += va * 0.25 + (vc / 15000).clip(upper=2.0)

    # Exclude history movies
    history_mask = TOP_MOVIES["tmdb_id"].astype(int).isin(history_ids)
    score[history_mask] = -1.0

    df = TOP_MOVIES.copy()
    df["_score"] = score
    candidates = df[df["_score"] >= 0].nlargest(n, "_score")

    # Fallback: pad with top-rated unseen movies if needed
    if len(candidates) < 5:
        seen_ids = set(candidates["tmdb_id"].astype(int)) | history_ids
        fallback = (
            df[~df["tmdb_id"].astype(int).isin(seen_ids)]
            .nlargest(n - len(candidates), "vote_count")
        )
        candidates = pd.concat([candidates, fallback])

    return candidates.head(n).reset_index(drop=True)


def _format_candidate(row: pd.Series) -> str:
    genres = str(row.get("genres") or "").strip()
    overview = str(row.get("overview") or "").strip()[:100]
    director = str(row.get("director") or "").strip()
    va = row.get("vote_average")
    rating = f"{float(va):.1f}/10" if pd.notna(va) and va else ""
    year = int(row.get("year") or 0)
    return (
        f'[{int(row["tmdb_id"])}] "{row["title"]}" ({year}) {rating} | {genres} | {director} | {overview}'
    )


# Two brief few-shot examples showing the target description style.
_FEW_SHOT = """EXAMPLES (style guide only):
Prefs: "psychological thrillers with twists" → "Nolan bends reality itself — a crew dives into dreams-within-dreams with stakes that escalate every minute. If mind-bending twists are your thing, the ending will haunt you for days."
Prefs: "funny feel-good films" → "Wes Anderson's wittiest caper: a legendary concierge caught in a hilarious murder mystery. Ralph Fiennes is a comedic revelation — gorgeous, laugh-out-loud, impossible not to love."
"""


def build_prompt(
    preferences: str,
    history: list[str],
    history_ids: list[int],
    candidates: pd.DataFrame,
) -> str:
    history_text = (
        "; ".join(f'"{n}" (id={i})' for n, i in zip(history, history_ids))
        if history
        else "none"
    )
    movie_list = "\n\n".join(_format_candidate(row) for _, row in candidates.iterrows())

    return f"""You are a film critic. Pick the best movie from the list for this user and write a compelling pitch.

{_FEW_SHOT}
USER PREFERENCES: "{preferences}"
ALREADY WATCHED (never recommend): {history_text}

CANDIDATES (pick exactly one tmdb_id):
{movie_list}

Write a description that directly references the user's preferences, under 500 chars.
Output ONLY JSON: {{"tmdb_id": <integer>, "description": "<pitch under 500 chars>"}}"""


def call_llm(prompt: str) -> dict:
    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )
    for attempt in range(2):
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        content = response.message.content.strip()
        if not content:
            continue
        try:
            result = json.loads(content)
            if "tmdb_id" in result:
                result["tmdb_id"] = int(result["tmdb_id"])
            if "description" in result:
                result["description"] = str(result["description"])[:500]
            return result
        except json.JSONDecodeError:
            if attempt == 1:
                raise
    raise ValueError("LLM returned empty response after 2 attempts")


def _fallback_response(candidates: pd.DataFrame, history_id_set: set) -> dict:
    """Build a safe response from top candidate without calling the LLM."""
    for _, row in candidates.iterrows():
        tid = int(row["tmdb_id"])
        if tid in history_id_set:
            continue
        overview = str(row.get("overview") or "").strip()
        desc = overview[:497] + "..." if len(overview) > 500 else overview
        return {"tmdb_id": tid, "description": desc}
    # Should never reach here — candidates always has unseen movies
    raise RuntimeError("No valid candidate available")


def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    history_id_set = {int(i) for i in history_ids}

    candidates = _select_candidates(preferences, history_id_set, n=8)
    prompt = build_prompt(preferences, history, history_ids, candidates)

    try:
        result = call_llm(prompt)
    except Exception:
        return _fallback_response(candidates, history_id_set)

    # Validate: returned tmdb_id must be in the full database and not in history
    all_valid_ids = set(TOP_MOVIES["tmdb_id"].astype(int))
    tid = result.get("tmdb_id")
    if tid not in all_valid_ids or tid in history_id_set:
        result["tmdb_id"] = int(candidates.iloc[0]["tmdb_id"])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local movie recommendation test.")
    parser.add_argument("--preferences", type=str)
    parser.add_argument("--history", type=str, help='Comma-separated titles, e.g. "The Avengers, Up"')
    args = parser.parse_args()

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    print(result)
    elapsed = time.perf_counter() - start
    print(f"\nServed in {elapsed:.2f}s")
