"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
from utils import match_movie_title


def recommend_random(movies, user_rating, k=5):
    """
    return k random unseen movies for user 
    """

    return random_movies


def recommend_most_popular(user_rating, movies, ratings, k=5):
    """
    return k movies from list of 50 best rated movies unseen for user
    """

    return popular movies


def recommend_from_same_cluster(user_rating, movies, k=3):
    """
    Return k most similar movies to the one spicified in the movieID
    
    INPUT
    - user_rating: a dictionary of titles and ratings
    - movies: a data frame with movie titles and cluster number
    - k: number of movies to recommend

    OUTPUT
    - title: the matched movie title (with fuzzy wuzzy) of the best ranked entry
    - movie_titles 
    """

    return best_rated_title, movie_titles



def recommend_with_NMF(user_item_matrix, user_rating, model, k=5):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model

    OUTPUT
    - a list of movieIds
    """
    
    # initialization - impute missing values    
    
    # transform user vector into hidden feature space
    
    # inverse transformation

    # build a dataframe

    # discard seen movies and sort the prediction
    
    # return a list of movie ids
    pass

def recommend_with_user_similarity(user_item_matrix, user_rating, k=5):
    pass


def similar_movies(movieId, movie_movie_distance_matrix):
    pass
