import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_ratings(fname):
    ratings = pd.read_csv(fname, delimiter='::', names=['userId', 'movieId', 'rating', 'timestamp'])
    
    indices = range(len(ratings))
    train_val_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, random_state=42)
    
    movie_idxs = {}
    user_idxs = {}
    def get_user_idx(user_id):
        if not user_id in user_idxs:
            user_idxs[user_id] = len(user_idxs)
        return user_idxs[user_id]
    
    def get_movie_idx(movie_id):
        if not movie_id in movie_idxs:
            movie_idxs[movie_id] = len(movie_idxs)
        return movie_idxs[movie_id]    

    num_users = ratings.userId.nunique()
    num_movies = ratings.movieId.nunique()
    data = {
        'train': {
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
        },
        'val': {
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
        },
        'test': {
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
        },
    }

    for indices, k in [(train_indices, 'train'), (val_indices, 'val'), (test_indices, 'test')]:
        for row in ratings.iloc[indices].itertuples():
            user_idx = get_user_idx(row.userId)
            movie_idx = get_movie_idx(row.movieId)
            data[k]['ratings'][user_idx, movie_idx] = row.rating
            data[k]['users'].add(user_idx)
            data[k]['movies'].add(movie_idx)
            
    return data
