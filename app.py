import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re

# -------------------------------------------------------------------
# Simple preprocessing (no NLTK)
# -------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    # keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    # we don't manually remove stopwords; TfidfVectorizer will do that
    return text

# -------------------------------------------------------------------
# Load data and build similarity matrix
# -------------------------------------------------------------------
@st.cache_resource
def load_data_and_model():
    # movies.csv must be in the same repo directory as app.py on Streamlit Cloud
    df = pd.read_csv("movies.csv")

    # Ensure required columns exist, fill missing
    for col in ["genres", "keywords", "overview"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    df["combined"] = df["genres"] + " " + df["keywords"] + " " + df["overview"]

    data = df[["title", "combined"]].copy()

    # Clean the text
    data["cleaned_text"] = data["combined"].apply(preprocess_text)

    # TF-IDF and cosine similarity
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"   # built-in English stopwords, no NLTK
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(data["cleaned_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, data, cosine_sim

df, data, cosine_sim = load_data_and_model()

# -------------------------------------------------------------------
# Recommendation function
# -------------------------------------------------------------------
def recommend_movies(movie_name: str, top_n: int = 5):
    idx = data[data["title"].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return None
    idx = idx[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the first (itself)
    sim_scores = sim_scores[1 : top_n + 1]

    movie_indices = [i for i, _ in sim_scores]
    recs = data.iloc[movie_indices].copy()
    recs["similarity"] = [score for _, score in sim_scores]
    return recs[["title", "similarity"]]

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write(
    "Type a movie name or pick one from the dropdown, and I’ll recommend similar movies "
    "based on genres, keywords, and overview using TF-IDF + cosine similarity."
)

st.subheader("1️⃣ Choose a movie")

col1, col2 = st.columns(2)

with col1:
    selected_title = st.selectbox(
        "Pick a movie from the list:",
        options=sorted(data["title"].dropna().unique()),
        index=0,
    )

with col2:
    manual_title = st.text_input(
        "…or type a movie name:",
        value="",
        placeholder="e.g., Avatar",
    )

movie_input = manual_title.strip() if manual_title.strip() else selected_title

st.subheader("2️⃣ Number of recommendations")
top_n = st.slider(
    "How many similar movies do you want?",
    min_value=3,
    max_value=20,
    value=5,
    step=1,
)

st.subheader("3️⃣ Get recommendations")

if st.button("Recommend 🎥"):
    with st.spinner("Finding similar movies..."):
        recs = recommend_movies(movie_input, top_n=top_n)

    if recs is None:
        st.error("❌ Movie not found in the dataset. Try picking from the dropdown.")
    else:
        st.success(f"Here are {len(recs)} movies similar to **{movie_input}**:")
        st.dataframe(recs.reset_index(drop=True))

        original_row = df[df["title"].str.lower() == movie_input.lower()]
        if not original_row.empty and "overview" in original_row.columns:
            st.markdown("---")
            st.markdown("### About the selected movie")
            st.markdown(f"**Title:** {original_row.iloc[0]['title']}")
            st.markdown(f"**Overview:** {original_row.iloc[0]['overview']}")
