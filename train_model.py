import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import ast

def load_data(filepath, limit=10000):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    # Sort by popularity or vote_count to get the best movies
    if 'popularity' in df.columns:
        df = df.sort_values('popularity', ascending=False)
    
    # Take top `limit` records to prevent memory overflow
    df = df.head(limit).copy()
    print(f"Loaded {len(df)} movies.")
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # Select important columns
    # 'cast' was not in the head output but let's check if it exists, otherwise just use what is there.
    features = ['genres', 'keywords', 'overview']
    if 'cast' in df.columns:
        features.append('cast')
        
    for feature in features:
        df[feature] = df[feature].fillna('')
        
    def combine_features(row):
        combined = []
        for feature in features:
            val = str(row[feature])
            # If it looks like a list or dict string from TMDB, just clean it simply or just add as is
            # For simplicity let's just combine the strings as text
            combined.append(val)
        return " ".join(combined)

    df['tags'] = df.apply(combine_features, axis=1)
    
    # Lowercase text
    df['tags'] = df['tags'].apply(lambda x: x.lower())
    
    # Keep only required columns
    # Let's keep id, title, tags, and perhaps some others if useful for UI, but title and final_cluster are primary
    final_df = df[['id', 'title', 'tags', 'poster_path']].copy()
    final_df['poster_path'] = final_df['poster_path'].fillna('')
    final_df.reset_index(drop=True, inplace=True)
    return final_df

def apply_clustering(df):
    print("Vectorizing text...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    
    n_clusters = 50 # Adjust as needed
    
    print("Applying KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(vectors)
    
    print("Applying DBSCAN...")
    dbscan = DBSCAN(eps=0.5, min_samples=3, metric='cosine')
    dbscan_labels = dbscan.fit_predict(vectors)
    
    print("Applying Agglomerative Clustering...")
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(vectors)
    
    # Combine labels using a string representation of the tuple
    # This forms a unique signature for each combination of clusters
    print("Combining clusters...")
    df['kmeans_cluster'] = kmeans_labels
    df['dbscan_cluster'] = dbscan_labels
    df['agg_cluster'] = agg_labels
    
    df['final_cluster'] = df.apply(
        lambda row: f"{row['kmeans_cluster']}_{row['dbscan_cluster']}_{row['agg_cluster']}", 
        axis=1
    )
    
    print("Clustering complete.")
    return df

if __name__ == "__main__":
    filepath = "TMDB_movie_dataset_v11.csv"
    try:
        df = load_data(filepath, limit=10000)
        processed_df = preprocess_data(df)
        clustered_df = apply_clustering(processed_df)
        
        # Save final dataframe
        output_file = "final_clustered_movies.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(clustered_df, f)
            
        print(f"Successfully saved clustered dataset to {output_file}")
    except Exception as e:
        print(f"Error during training: {e}")
