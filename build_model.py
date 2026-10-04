"""Build the content-based recommendation model from the TMDB 5000 dataset.

Creates two files used by app.py:
- movies.pkl:     DataFrame with movie id and title
- similarity.pkl: cosine-similarity matrix between all movies
"""

import ast
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "tmdb_5000_movies.csv"


def parse_names(json_text, limit=None):
    """Turn '[{"id": 28, "name": "Action"}, ...]' into ['Action', ...]."""
    names = [item["name"] for item in ast.literal_eval(json_text)]
    return names[:limit] if limit else names


def collapse(names):
    """Join multi-word names into single tokens, e.g. 'Science Fiction' -> 'sciencefiction'."""
    return [name.replace(" ", "").lower() for name in names]


def build_tags(movies):
    """Combine genres, keywords, production companies, overview and tagline into one text field."""
    genres = movies["genres"].apply(parse_names).apply(collapse)
    keywords = movies["keywords"].apply(parse_names).apply(collapse)
    companies = movies["production_companies"].apply(lambda t: parse_names(t, limit=3)).apply(collapse)
    overview = movies["overview"].fillna("").str.lower().str.split()
    tagline = movies["tagline"].fillna("").str.lower().str.split()

    return (genres + keywords + companies + overview + tagline).apply(" ".join)


def build_model():
    movies = pd.read_csv(DATA_PATH)
    movies = movies.dropna(subset=["title"]).reset_index(drop=True)

    movies["tags"] = build_tags(movies)

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(movies["tags"])
    similarity = cosine_similarity(vectors).astype("float32")

    return movies[["id", "title"]], similarity


if __name__ == "__main__":
    movies_df, similarity = build_model()

    with open(BASE_DIR / "movies.pkl", "wb") as f:
        pickle.dump(movies_df, f)
    with open(BASE_DIR / "similarity.pkl", "wb") as f:
        pickle.dump(similarity, f)

    print(f"Model built: {len(movies_df)} movies, similarity matrix {similarity.shape}")
