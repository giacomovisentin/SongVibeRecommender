import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder="templates")

# === Load data ===
EMB_PATH = "embeddings.npy"
META_PATH = "song_metadata.parquet"

if not os.path.exists(EMB_PATH):
    raise FileNotFoundError(f"Missing {EMB_PATH}")
if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Missing {META_PATH}")

emb = np.load(EMB_PATH)
X = pd.read_parquet(META_PATH).reset_index(drop=True)
GENRE_COLS = [c for c in X.columns if c.startswith("genre_")]

# Check for track_id, essential for Spotify links
if "track_id" not in X.columns:
    raise ValueError("FATAL: 'track_id' not found in song_metadata.parquet. Please re-run data saving script.")

# Lowercase columns for quick substring search
X["_track_lower"] = X["track_name"].fillna("").str.lower()
X["_artist_lower"] = X["artist_name"].fillna("").str.lower()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/suggest")
def suggest():
    """Return songs containing the query in title or artist"""
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])

    mask = X["_track_lower"].str.contains(q) | X["_artist_lower"].str.contains(q)
    matches = X[mask].head(20)
    results = [
        {
            "index": int(i),
            "track_id": r["track_id"], # Send track_id
            "track_name": r["track_name"],
            "artist_name": r["artist_name"]
        }
        for i, r in matches.iterrows()
    ]
    return jsonify(results)

@app.route("/recommend", methods=["POST"])
def recommend():
    """Return recommendations for a selected song index"""
    data = request.get_json(force=True)
    idx = int(data.get("index", -1))
    if idx < 0 or idx >= len(X):
        return jsonify({"status": "error", "message": "Invalid song index."}), 400

    sims = cosine_similarity(emb[idx].reshape(1, -1), emb).flatten()
    order = sims.argsort()[::-1][1:11]  # skip itself
    recs = X.iloc[order].copy()
    recs["similarity"] = sims[order]

    def extract_genres(row):
        genres = [g.replace("genre_", "") for g in GENRE_COLS if row.get(g, 0) > 0]
        return ", ".join(genres) if genres else "N/A"

    recs["genres"] = recs.apply(extract_genres, axis=1)

    # Send track_id for recommendations
    results = recs[["track_id", "track_name", "artist_name", "genres", "similarity"]].to_dict(orient="records")
    
    selected_row = X.loc[idx]
    selected = {
        "track_id": selected_row["track_id"], # Send track_id for selected song
        "track_name": selected_row["track_name"],
        "artist_name": selected_row["artist_name"]
    }
    return jsonify({"status": "success", "selected": selected, "recommendations": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)