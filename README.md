# Movie Recommender

An agentic movie recommendation system that uses a two-stage approach to select and pitch the best movie for each user from a database of 1,000 popular TMDB films.

---

## Approach

### Stage 1 — Smart Candidate Selection

Rather than sending all 1,000 movies to the LLM, the agent first narrows the pool to the 15 most relevant candidates using a scoring function (`_select_candidates` in [llm.py](llm.py)):

- **Genre/theme alignment**: user preference text is matched against a keyword trigger map covering 16 genres (Action, Comedy, Horror, Superhero, Buddy, etc.). Each genre match awards a strong relevance bonus.
- **Token-level matching**: content words from the user's preferences are searched in each movie's genres, keywords, overview, and cast fields.
- **Quality signal**: vote average and vote count contribute a smaller bonus, favouring well-regarded films when genre matches are equal.
- **History filtering**: movies in the user's watch history are excluded before scoring.

A fallback path pads the candidate list with top-rated unseen movies when preferences are unusual or very specific.

### Stage 2 — LLM Selection and Description

The top 15 candidates are passed to `gemma4:31b-cloud` via the Ollama API with rich per-movie metadata: genres, rating, director, top 4 cast members, keywords, and a 220-character overview excerpt.

The prompt instructs the LLM to:
1. Pick the single best match for the user's stated preferences.
2. Write a personalised pitch (≤500 characters) that is exciting, specific to the user's tastes, and avoids spoilers.

### Validation and Fallback

After the LLM responds, the code validates that the returned `tmdb_id` exists in the full database and is not in the user's history. If either check fails, the highest-scored candidate from Stage 1 is substituted automatically.

---

## Evaluation Strategy

The agent is evaluated by running `test.py`, which checks:

- `tmdb_id` is a valid integer present in `TOP_MOVIES` (the full 1,000-movie database).
- `tmdb_id` is not in the user's watch history.
- Both keys (`tmdb_id`, `description`) are present in the returned dict.
- The full response is returned within 20 seconds.

Beyond automated checks, recommendation quality was assessed manually by inspecting whether Stage 1 surfaces intuitively correct candidates for a variety of preference strings (e.g. "superheroes and buddy cop", "funny and feel-good", "psychological thriller"), and checking that the LLM's final pick and description feel personal and compelling rather than generic.

---

## Code Guide

| File | Purpose |
|---|---|
| [llm.py](llm.py) | Main implementation — scoring, candidate selection, prompt construction, LLM call |
| [test.py](test.py) | Automated test suite — run this before submitting |
| [tmdb_top1000_movies.csv](tmdb_top1000_movies.csv) | Movie database (1,000 films, 24 columns) |
| `requirements.txt` | Python dependencies |

Key functions in [llm.py](llm.py):

| Function | Description |
|---|---|
| `_score_movie` | Computes a relevance score for one movie row against user preferences |
| `_select_candidates` | Scores all 1,000 movies and returns the top N, excluding history |
| `_format_candidate` | Formats a movie row as a readable string for the prompt |
| `build_prompt` | Assembles the full LLM prompt from candidates and user input |
| `call_llm` | Sends the prompt to `gemma4:31b-cloud` and parses the JSON response |
| `get_recommendation` | Top-level function called once per request |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at [ollama.com/settings/keys](https://ollama.com/settings/keys).

---

## Running

**Interactive:**
```bash
OLLAMA_API_KEY=your_key_here python llm.py
# Preferences: I love sci-fi thrillers
# Watch history (optional): Inception
```

**With flags:**
```bash
OLLAMA_API_KEY=your_key_here python llm.py \
  --preferences "I want a funny, light, action-packed movie." \
  --history "The Dark Knight Rises"
```

**Test suite:**
```bash
OLLAMA_API_KEY=your_key_here python test.py
```

---

## API Contract

```python
def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
```

| Argument | Type | Description |
|---|---|---|
| `preferences` | `str` | Free-text description of what the user wants to watch |
| `history` | `list[str]` | Movie titles the user has already seen |
| `history_ids` | `list[int]` | TMDB IDs corresponding to `history` |

Returns a `dict`:

| Key | Type | Description |
|---|---|---|
| `tmdb_id` | `int` | A valid TMDB ID from the 1,000-movie database |
| `description` | `str` | Personalised pitch, ≤500 characters |

**Do NOT hard-code your API key.** The grader injects `OLLAMA_API_KEY` at run time:

```python
os.environ["OLLAMA_API_KEY"]   # read from environment, not from source
```

---

## Submission

Submit a zip containing at minimum:

- `llm.py`
- `requirements.txt`
- `tmdb_top1000_movies.csv`

Do not include `.env`, `.venv/`, or `__pycache__`. Keep the zip under 10 MB.
