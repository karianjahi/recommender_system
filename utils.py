"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from sklearn.impute import SimpleImputer
import pickle


movies = pd.read_csv('data/movies_clusters_ratings.csv')
ratings = pd.read_csv("data/ratings_clean.csv")
user_item_matrix = pd.read_csv("data/my_user_item_matrix.csv", index_col=0)
model = pickle.load(open("nmf_model.sav","rb"))


def match_movie_title(input_title, movie_titles):
    """
    Matches inputed movie title to existing one in the list with fuzzywuzzy
    """
    matched_title = process.extractOne(input_title, movie_titles)[0]

    return matched_title

def print_movie_titles(movie_titles):
    """
    Prints list of movie titles in cli app
    """    
    pass


def create_user_vector(user_rating, movies):
    """
    Convert dict of user_ratings to a user_vector
    """       
    # generate the user vector
    print(user_rating)
    user_vector = None
    return user_vector

def lookup_movieId(movieId):
    """
    Convert output of recommendation to movie title
    """
    assert isinstance(movieId, int)
    # match movieId to title
    movie_title = list(movies[movies["movieid"] == movieId]["title"])[0]
    return movie_title

def create_user_item_matrix():
    pass


def printt(item):
    print("=======================================")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print(item)
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("=======================================")




if __name__ == "__main__":
    user_rating = {
        "four rooms": 5,
        "sudden death": 3,
        "othello": 4,
        "nixon": 3,
        "Golden eye": 1,
        "total eclipse": 5,
        "nadja": 3
    }


    #print(create_user_vector(user_rating, movies))
    print("====================")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print(user_item_matrix)