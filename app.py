import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- 1. Inizializzazione e Caricamento Dati ---
app = Flask(__name__)

EMBEDDINGS_PATH = "embeddings.npy"
METADATA_PATH = "song_metadata.parquet"

print("Caricamento embeddings...")
try:
    emb = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings caricati. Shape: {emb.shape}")
except Exception as e:
    print(f"ERRORE nel caricamento di '{EMBEDDINGS_PATH}': {e}")

print("Caricamento metadati...")
try:
    X = pd.read_parquet(METADATA_PATH)
    print(f"Metadati caricati. Righe: {len(X)}")
except Exception as e:
    print(f"ERRORE nel caricamento di '{METADATA_PATH}': {e}")

GENRE_COLS = [col for col in X.columns if col.startswith('genre_')]
print("Server pronto.")

# --- 2. Logica di Raccomandazione (Identica a prima) ---

def find_song_index(title, artist):
    matches = X[
        (X['track_name'].str.lower() == title.lower()) &
        (X['artist_name'].str.lower() == artist.lower())
    ]
    if matches.empty:
        raise ValueError(f"Canzone non trovata: '{title}' di '{artist}'")
    return matches.index[0]

def get_recommendations(target_index, top_k=10):
    target_embedding = emb[target_index].reshape(1, -1)
    similarity_scores = cosine_similarity(target_embedding, emb).flatten()
    sorted_indices = similarity_scores.argsort()[::-1][1:top_k+1]
    recs_df = X.iloc[sorted_indices].copy()
    
    def extract_genres(row):
        genres = [col.replace('genre_', '') for col in GENRE_COLS if row[col] > 0]
        return ', '.join(genres) or "N/A"
    
    recs_df['genres'] = recs_df.apply(extract_genres, axis=1)
    recs_df['similarity'] = similarity_scores[sorted_indices]
    
    return recs_df[['track_name', 'artist_name', 'genres', 'similarity']]

# --- 3. Endpoint API (Identici a prima) ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        title = data['title']
        artist = data['artist']
        if not title or not artist:
            raise ValueError("Titolo e Artista sono richiesti.")
        target_index = find_song_index(title, artist)
        recs_df = get_recommendations(target_index)
        return jsonify({"status": "success", "recommendations": recs_df.to_dict('records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# --- 4. Avvio del Server (MODIFICATO PER RENDER) ---
if __name__ == '__main__':
    # Render imposterà la variabile PORT. 
    # '0.0.0.0' dice a Flask di essere accessibile dall'esterno.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)