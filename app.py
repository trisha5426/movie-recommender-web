"""Movie recommender web app (content-based, TMDB 5000).
Method and logic adapted from https://github.com/Mehak2005si/movie-recommender by Mehak Singh.
"""
import ast
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def names(obj, limit=None):
    out = [i["name"] for i in ast.literal_eval(obj)]
    return out[:limit] if limit else out


def director(obj):
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            return [i["name"]]
    return []


@st.cache_resource(show_spinner="Loading data and building similarity matrix...")
def build():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]].dropna().copy()
    movies["genres"] = movies["genres"].apply(names)
    movies["keywords"] = movies["keywords"].apply(names)
    movies["cast"] = movies["cast"].apply(lambda x: names(x, 3))
    movies["crew"] = movies["crew"].apply(director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())
    for c in ["genres", "keywords", "cast", "crew"]:
        movies[c] = movies[c].apply(lambda x: [i.replace(" ", "") for i in x])
    tags = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    df = movies[["title"]].copy()
    df["tags"] = tags.apply(lambda x: " ".join(x).lower())
    df = df.reset_index(drop=True)
    vectors = CountVectorizer(max_features=5000, stop_words="english").fit_transform(df["tags"])
    return df, cosine_similarity(vectors)


def recommend(df, sim, title):
    idx = df[df["title"] == title].index[0]
    top = sorted(enumerate(sim[idx]), reverse=True, key=lambda x: x[1])[1:6]
    return [df.iloc[i].title for i, _ in top]


st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender")
st.caption("Content-based: movies with similar overview, genres, keywords, top 3 cast and director (TMDB 5000).")
df, sim = build()
choice = st.selectbox("Pick a movie", sorted(df["title"].unique()), index=None, placeholder="Type to search...")
if choice:
    st.subheader(f"Because you picked {choice}:")
    for n, t in enumerate(recommend(df, sim, choice), 1):
        st.write(f"{n}. {t}")
st.markdown("---")
st.caption("Text similarity only (CountVectorizer + cosine). Based on github.com/Mehak2005si/movie-recommender by Mehak Singh. Data: TMDB 5000 via Kaggle.")
st.caption("This application uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise approved by TMDB.")
