import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# -----------------------------------------------------------------------------
# NLTK setup (runs once; cached by Streamlit)
# -----------------------------------------------------------------------------
@st.cache_resource
def init_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
    return set(stopwords.words("english"))

stop_words = init_nltk()

# -----------------------------------------------------------------------------
# Text preprocessing (same idea as your notebook)
# -----------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# -----------------------------------------------------------------------------
# Load data and build similarity matrix
# -----------------------------------------------------------------------------
@st.cache_resource
def load_data_and_model():
    # ⚠️ Make sure movies.csv is in the same folder as this app.py
    df = pd.read_csv("movies.csv")

    # Mirror your notebook logic
    # Safely handle missing values
    for col in ["genres", "keywords", "overview"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""  # in case some column is missing

    df["combined"] = df["genres"] + " " + df["keywords"] + " " + df["overview"]

    # Keep just what we need
    data = df[["title", "combined"]].copy()

    # Clean the text
    data["cleaned_text"] = data["combined"].apply(preprocess_text)

    # TF-IDF and cosine similarity
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data["cleaned_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Return everything we might want
    return df, data, cosine_sim

df, data, cosine_sim = load_data_and_model()

# -----------------------------------------------------------------------------
# Recommendation function (adapted from your notebook)
# -----------------------------------------------------------------------------
def recommend_movies(movie_name: str, top_n: int = 5):
    # Find the index of the movie (case-insensitive)
    idx = data[data["title"].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return None  # indicate "not found"
    idx = idx[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the first one (it's the movie itself)
    sim_scores = sim_scores[1 : top_n + 1]

    # Get movie indices
    movie_indices = [i for i, _ in sim_scores]

    # Return titles (and optionally similarity scores)
    recs = data.iloc[movie_indices].copy()
    recs["similarity"] = [score for _, score in sim_scores]
    return recs[["title", "similarity"]]

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write(
    "Type a movie name or pick one from the dropdown, and I'll suggest similar movies "
    "based on genres, keywords, and overview using TF-IDF + cosine similarity."
)

# Choose movie either via dropdown or text
st.subheader("1️⃣ Choose a movie")

col1, col2 = st.columns(2)

with col1:
    # Dropdown with all titles
    selected_title = st.selectbox(
        "Pick a movie from the list:",
        options=sorted(data["title"].unique()),
        index=0,
    )

with col2:
    manual_title = st.text_input(
        "…or type a movie name:",
        value="",
        placeholder="e.g., Avatar",
    )

# If user typed something, use that; otherwise use dropdown value
if manual_title.strip():
    movie_input = manual_title.strip()
else:
    movie_input = selected_title

st.subheader("2️⃣ Number of recommendations")
top_n = st.slider("How many similar movies do you want?", min_value=3, max_value=20, value=5, step=1)

st.subheader("3️⃣ Get recommendations")
if st.button("Recommend 🎥"):
    with st.spinner("Finding similar movies..."):
        recs = recommend_movies(movie_input, top_n=top_n)

    if recs is None:
        st.error("❌ Movie not found in the dataset. Try picking from the dropdown instead.")
    else:
        st.success(f"Here are {len(recs)} movies similar to **{movie_input}**:")
        # Show as table
        st.dataframe(recs.reset_index(drop=True))

        # Optional: show the overview of the selected movie
        original_row = df[df["title"].str.lower() == movie_input.lower()]
        if not original_row.empty:
            st.markdown("---")
            st.markdown("### About the selected movie")
            st.markdown(f"**Title:** {original_row.iloc[0]['title']}")
            if "overview" in original_row.columns:
                st.markdown(f"**Overview:** {original_row.iloc[0]['overview']}")