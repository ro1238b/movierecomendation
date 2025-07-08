import pickle
import difflib
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load preprocessed data
# -------------------------------
df = pickle.load(open(r'F:\movie recommendation 2\movies.pkl', 'rb'))
similarity = pickle.load(open(r'F:\movie recommendation 2\similarity.pkl', 'rb'))

# -------------------------------
# Load BERT model
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# Prepare movie title embeddings for suggestions
# -------------------------------
movie_titles = df['title'].dropna().tolist()
title_embeddings = model.encode(movie_titles, show_progress_bar=False)

# -------------------------------
# TMDB API Key
# -------------------------------
TMDB_API_KEY = '0d260096299be58b578bb54a10a9b073'

# -------------------------------
# Fetch movie poster + overview
# -------------------------------
def fetch_movie_details_by_id(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        poster_path = data.get('poster_path')
        overview = data.get('overview') or "No description available."

        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "static/placeholder.jpg"
        return poster_url, overview

    except Exception as e:
        print(f"TMDB API error for ID {movie_id}:", e)
        return "static/placeholder.jpg", "Error fetching movie info."


# -------------------------------
# Recommend similar movies
# -------------------------------
def recommend(movie):
    movie = movie.strip().lower()
    titles = df['title'].str.lower()

    close_matches = difflib.get_close_matches(movie, titles.values, n=1, cutoff=0.6)
    if not close_matches:
        return None, [], movie

    best_match = close_matches[0]
    index = titles[titles == best_match].index[0]

    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    recommended = []
    for i in distances[1:7]:
       movie_row = df.iloc[i[0]]
       poster, overview = fetch_movie_details_by_id(movie_row.id)  # or movie_row.movie_id
       title = movie_row.title
       recommended.append({
            'title': title,
            'poster_url': poster,
            'overview': overview
        })

    return best_match.title(), recommended, movie

# -------------------------------
# Suggest closest movie names (search box)
# -------------------------------
def get_suggestions(query, limit=5):
    if not query:
        return []

    query_vec = model.encode([query])
    similarities = cosine_similarity(query_vec, title_embeddings)[0]
    top_indices = similarities.argsort()[-limit:][::-1]

    return [movie_titles[i] for i in top_indices]
