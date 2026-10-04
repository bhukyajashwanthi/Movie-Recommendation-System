import pickle
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MOVIES_PATH = BASE_DIR / "movies.pkl"
SIMILARITY_PATH = BASE_DIR / "similarity.pkl"
NUM_RECOMMENDATIONS = 5


@st.cache_resource
def load_model():
    """Load the movies DataFrame and similarity matrix once per session."""
    with open(MOVIES_PATH, "rb") as f:
        movies_df = pickle.load(f)
    with open(SIMILARITY_PATH, "rb") as f:
        similarity = pickle.load(f)
    return movies_df, similarity


def recommend(movie, movies_df, similarity, n=NUM_RECOMMENDATIONS):
    """Return the n movies most similar to the selected one."""
    movie_index = movies_df[movies_df["title"] == movie].index[0]
    distances = similarity[movie_index]

    # Sort by similarity score and skip the first result (the movie itself)
    recommended_indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)[1 : n + 1]

    return [movies_df.iloc[i].title for i in recommended_indices]


# Streamlit app setup
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.write("Pick a movie you like and get 5 similar movies, based on genres, keywords, plot and production studios.")

if not MOVIES_PATH.exists() or not SIMILARITY_PATH.exists():
    st.error("Model files not found. Run `python build_model.py` first to create them.")
    st.stop()

movies_df, similarity = load_model()

selected_movie_name = st.selectbox("Choose a movie you like:", movies_df["title"].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie_name, movies_df, similarity)
    st.subheader(f"Because you liked {selected_movie_name}:")
    for rank, title in enumerate(recommendations, start=1):
        st.write(f"{rank}. {title}")
