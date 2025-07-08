import pandas as pd
import numpy as np
import ast
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Step 1: Load the raw CSV files ------------------
movies = pd.read_csv(r'F:\movie recommendation 2\tmdb_5000_movies.csv')
credits = pd.read_csv(r'F:\movie recommendation 2\tmdb_5000_credits.csv')

# ------------------ Step 2: Define functions to extract info ------------------
def extract_names(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)]
    except Exception:
        return []

def get_top_cast(x, n=3):
    try:
        cast_list = ast.literal_eval(x)
        return [i['name'] for i in cast_list[:n]]
    except Exception:
        return []

def get_director(x):
    try:
        crew_list = ast.literal_eval(x)
        for member in crew_list:
            if member.get('job') == 'Director':
                return member.get('name', '')
        return ''
    except Exception:
        return ''

def clean_text(x):
    return ' '.join(x) if isinstance(x, list) else x

# ------------------ Step 3: Process the columns ------------------
for col in ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']:
    movies[col] = movies[col].apply(lambda x: extract_names(x) if isinstance(x, str) else [])

credits['cast'] = credits['cast'].apply(lambda x: get_top_cast(x) if isinstance(x, str) else [])
credits['director'] = credits['crew'].apply(lambda x: get_director(x) if isinstance(x, str) else '')

# ------------------ Step 4: Merge DataFrames (Preserve 'id') ------------------
# ⚠️ Only select needed columns, including 'id'
movies_small = movies[['id', 'title', 'genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']]
credits_small = credits[['title', 'cast', 'director']]

df = movies_small.merge(credits_small, on='title')

# ------------------ Step 5: Create the 'tags' column ------------------
df['tags'] = df['genres'].apply(clean_text) + ' ' + \
             df['keywords'].apply(clean_text) + ' ' + \
             df['cast'].apply(clean_text) + ' ' + \
             df['director'].apply(lambda x: x if isinstance(x, str) else '')

df['tags'] = df['tags'].str.lower().fillna('')

# ------------------ Step 6: Generate Sentence-BERT embeddings ------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating BERT embeddings for content...")

embeddings = model.encode(df['tags'].tolist(), show_progress_bar=True)
similarity = cosine_similarity(embeddings)

movie_titles = df['title'].dropna().tolist()
title_embeddings = model.encode(movie_titles, show_progress_bar=True)

# ------------------ Step 7: Save to Pickle Files ------------------
pickle.dump(df, open(r'F:\movie recommendation 2\movies.pkl', 'wb'))
pickle.dump(similarity, open(r'F:\movie recommendation 2\similarity.pkl', 'wb'))
pickle.dump(movie_titles, open(r'F:\movie recommendation 2\movie_titles.pkl', 'wb'))
pickle.dump(title_embeddings, open(r'F:\movie recommendation 2\title_embeddings.pkl', 'wb'))

print("\n✅ Data preparation complete. All files saved successfully.")
