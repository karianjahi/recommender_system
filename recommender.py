"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
from utils import match_movie_title, user_item_matrix, model, SimpleImputer, movies, lookup_movieId


def recommend_random(movies, user_rating, k=5):
    """
    return k random unseen movies for user 
    """

    return random_movies


def recommend_most_popular(user_rating, movies, ratings, k=5):
    """
    return k movies from list of 50 best rated movies unseen for user
    """

    return popular_movies


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
    imputer = SimpleImputer(strategy="constant", fill_value=2.5) 
    imputed_df = pd.DataFrame(imputer.fit_transform(user_item_matrix))
    imputed_df.columns = user_item_matrix.columns
    imputed_df.index = user_item_matrix.index
    
    # transform user vector into hidden feature space
    q_components = model.components_

    # transform user ratings into dataframe and fill all unrated movies

    # get all movie titles
    movie_ids_and_titles = movies[["movieid", "title"]]
    movie_collection_dict = movie_ids_and_titles.set_index("movieid").to_dict()["title"]
    movie_titles = list(movie_collection_dict.values())
    movie_ids = list(movie_collection_dict.keys())

    # use fuzzywuzzy to get the user intended movies
    intended_movies = [match_movie_title(i, movie_titles) for i in user_rating.keys()]
    
    # Get movie ids for the movies requested by user
    user_movie_ids_dict = {}
    for user_movie in intended_movies:
        for dict_movie_title, dict_movie_id in zip(movie_collection_dict.values(), movie_collection_dict.keys()):
            if user_movie == dict_movie_title:
                user_movie_ids_dict[user_movie] = dict_movie_id

    
    # interchange the user movie title with ids in the user query
    user_movie_ratings = {}
    for user_rating_key, user_movie_id in zip(user_rating, user_movie_ids_dict):
        user_movie_ratings[user_movie_ids_dict[user_movie_id]] = user_rating[user_rating_key]

    
    # Add unrated movie ids to the user rating
    user_rating_full = {}
    user_item_movie_ids = list(user_item_matrix.columns)
    for movieid in user_item_movie_ids:
        if movieid in user_movie_ratings.keys():
            user_rating_full[movieid] = user_movie_ratings[movieid]
        else:
            user_rating_full[movieid] = None
        
    # Convert user ratings full into a dataframe
    user_rating_table = pd.DataFrame(user_rating_full, index=[0])
    
    # fill in missing values with a constant
    user_rating_table.fillna(value=2.5, inplace=True)
    
    not_run = True
    # inverse transformation
    if not_run:
        q_df = pd.DataFrame(q_components)
        p_df = pd.DataFrame(model.transform(user_rating_table))

        # build a dataframe
        predicted_ratings = p_df.dot(q_df)

        # discard seen movies and sort the prediction
        predicted_ratings.drop(user_movie_ids_dict.values(), axis=1, inplace=True)

        predictions = predicted_ratings.T
        predictions.columns = ["ratings"]
        movie_ids_to_recommend = list(predictions.sort_values("ratings", ascending=False).head(k).index)
        
    # return a list of movie ids
    return movie_ids_to_recommend
    

def recommend_with_user_similarity(user_item_matrix, user_rating, k=5):
    pass


def similar_movies(movieId, movie_movie_distance_matrix):
    pass

if __name__ == "__main__":
    user_rating = {
        "four Rooms": 5,
        "sudden DEath": 3,
        "oThelo": 4,
        "Nixons": 3,
        "Golden eye": 1,
        "Total EclipseS": 5,
        "NadJa": 3,
        "forbiden planet": 4
    }
    
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
    print(recommend_with_NMF(user_item_matrix, user_rating, model, k=5))
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