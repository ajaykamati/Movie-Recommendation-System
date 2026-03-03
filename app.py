from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load the clustered movie data
MODEL_PATH = 'final_clustered_movies.pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        movies_df = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    movies_df = None

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
PLACEHOLDER_IMAGE = "https://via.placeholder.com/500x750?text=No+Poster+Available"

def get_poster_url(poster_path):
    if pd.isna(poster_path) or not str(poster_path).strip():
        return PLACEHOLDER_IMAGE
    return f"https://image.tmdb.org/t/p/w500{poster_path}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    if movies_df is None:
        return jsonify([])
    titles = movies_df['title'].tolist()
    return jsonify(titles)
    
@app.route('/popular', methods=['GET'])
def popular():
    if movies_df is None:
        return jsonify({"error": "Model not loaded."}), 500
        
    top_10 = movies_df.head(10)
    popular_movies = []
    for _, row in top_10.iterrows():
        title = row['title']
        poster_path = row.get('poster_path', '')
        popular_movies.append({
            "title": title,
            "poster": get_poster_url(poster_path)
        })
        
    return jsonify({"popular": popular_movies})

@app.route('/recommend', methods=['POST'])
def recommend():
    if movies_df is None:
        return jsonify({"error": "Model not loaded. Please ensure final_clustered_movies.pkl exists."}), 500
        
    movie_name = request.form.get('movie_name')
    if not movie_name:
        return jsonify({"error": "Please enter a movie name."}), 400
        
    # Find the movie
    # Using case-insensitive search
    match = movies_df[movies_df['title'].str.lower() == movie_name.lower()]
    
    if match.empty:
        # Try finding a partial match if exact match fails
        match = movies_df[movies_df['title'].str.contains(movie_name, case=False, na=False)]
        if match.empty:
            return jsonify({"error": "Movie not found in the database."}), 404
            
    # Get the cluster of the matched movie
    target_movie = match.iloc[0]
    cluster_label = target_movie['final_cluster']
    
    # Filter dataset for same cluster
    similar_movies = movies_df[movies_df['final_cluster'] == cluster_label]
    
    # Exclude the target movie itself
    similar_movies = similar_movies[similar_movies['id'] != target_movie['id']]
    
    if similar_movies.empty:
        return jsonify({"error": "No recommendations found for this movie."}), 404
        
    # We'll just take the top 5 from the cluster
    # They are already sorted by popularity (from preprocessing)
    top_5 = similar_movies.head(5)
    
    recommendations = []
    for _, row in top_5.iterrows():
        title = row['title']
        poster_path = row.get('poster_path', '')
        recommendations.append({
            "title": title,
            "poster": get_poster_url(poster_path)
        })
        
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
