# Movie Recommender Web App

A content-based movie recommender with a Streamlit web UI. Pick a movie and it suggests 5 similar movies, based on overview, genres, keywords, top 3 cast and director (CountVectorizer + cosine similarity over the TMDB 5000 dataset).

## Dataset (not included)

The two CSV files are NOT part of this repository. To run the app, download them yourself:

1. Get the TMDB 5000 Movie Dataset from Kaggle: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
2. Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the same folder as `app.py`.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501), pick a movie from the search box, and get 5 recommendations.

## Notes

- This repository contains code only; it is not a hosted app.
- This application uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise approved by TMDB.
