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
    overview = str(row.get("overview") or "").strip()[:220]
    director = str(row.get("director") or "").strip()
    cast = ", ".join(str(row.get("top_cast") or "").split(",")[:4]).strip()
    keywords = str(row.get("keywords") or "").strip()[:80]
    va = row.get("vote_average")
    rating = f"{float(va):.1f}/10" if pd.notna(va) and va else ""
    year = int(row.get("year") or 0)
    return (
        f'[tmdb_id={int(row["tmdb_id"])}] "{row["title"]}" ({year})\n'
        f"  Genres: {genres} | Rating: {rating} | Director: {director}\n"
        f"  Cast: {cast}\n"
        f"  Keywords: {keywords}\n"
        f"  Overview: {overview}"
    )


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

    return f"""You are an expert film critic and passionate movie recommender. Your goal: select the single best movie from the list below for this user and write a short, irresistible pitch that makes them genuinely excited to watch it.

USER PREFERENCES: "{preferences}"

ALREADY WATCHED — NEVER recommend these: {history_text}

CANDIDATE MOVIES — you MUST choose exactly one tmdb_id from this list:
{movie_list}

Selection criteria:
- Best match to the user's stated preferences (genre, themes, mood)
- High quality (good ratings, well-known cast/director)
- Something genuinely interesting to recommend

Description requirements (CRITICAL):
- Under 500 characters — violating this disqualifies the response
- Personal: explain why THIS user with THEIR preferences will love it
- Exciting: create urgency and enthusiasm
- Specific: mention a compelling detail (director, cast, theme) without spoiling plot twists

Output ONLY valid JSON, no markdown, no extra keys:
{{"tmdb_id": <integer from the list above>, "description": "<your pitch under 500 chars>"}}"""


def call_llm(prompt: str) -> dict:
    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    result = json.loads(response.message.content)
    if "tmdb_id" in result:
        result["tmdb_id"] = int(result["tmdb_id"])
    if "description" in result:
        result["description"] = str(result["description"])[:500]
    return result


def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    history_id_set = {int(i) for i in history_ids}

    candidates = _select_candidates(preferences, history_id_set, n=15)
    prompt = build_prompt(preferences, history, history_ids, candidates)
    result = call_llm(prompt)

    # Validate: returned tmdb_id must be in the full database and not in history
    all_valid_ids = set(TOP_MOVIES["tmdb_id"].astype(int))
    tid = result.get("tmdb_id")
    if tid not in all_valid_ids or tid in history_id_set:
        best = candidates.iloc[0]
        result["tmdb_id"] = int(best["tmdb_id"])

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
