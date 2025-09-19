# movie_recommender.py
# Simple Movie Recommendation System using cosine similarity

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
movies = {
    'title': [
        'The Matrix', 'The Matrix Reloaded', 'John Wick', 'Avengers: Endgame', 'Iron Man',
        'The Dark Knight', 'Batman Begins', 'Interstellar', 'Inception', 'Tenet'
    ],
    'description': [
        'A computer hacker learns about the true nature of reality',
        'Neo and friends continue the fight against the machines',
        'An ex-hitman seeks vengeance',
        'Heroes assemble to fight Thanos',
        'Tony Stark becomes Iron Man',
        'Batman fights Joker in Gotham',
        'Bruce Wayne becomes Batman',
        'A space mission to save humanity',
        'A thief enters dreams to steal secrets',
        'A secret agent manipulates time to prevent catastrophe'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(movies)

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def recommend_movie(title, num=3):
    if title not in df['title'].values:
        return ["Movie not found in dataset."]
    
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Test the recommender
print("Recommendations for 'The Matrix':")
print(recommend_movie('The Matrix'))
