"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep=',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas DataFrame
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

def calculate_cosine_similarity_matrix(data):
    """Calculate the cosine similarity matrix for the given movie data.

    Parameters
    ----------
    data : Pandas DataFrame
        Movie data.

    Returns
    -------
    numpy.ndarray
        Cosine similarity matrix.

    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['keyWords'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(movie_list, cosine_sim, indices, top_n=10):
    """Get top-n movie recommendations based on the given movie list.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    cosine_sim : numpy.ndarray
        Cosine similarity matrix.
    indices : pd.Series
        Movie indices.
    top_n : int, optional
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    recommended_movies = []
    idx_list = [indices[indices == movie].index[0] for movie in movie_list]
    
    # Calculate the average cosine similarity for the selected movies
    avg_cosine_sim = np.mean(cosine_sim[idx_list], axis=0)

    # Get the indexes of the 10 most similar movies
    top_indexes = np.argsort(avg_cosine_sim)[::-1][:top_n]

    # Store movie names
    recommended_movies = list(indices.iloc[top_indexes])
    return recommended_movies

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list, top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : int, optional
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(27000)
    
    # Calculate the cosine similarity matrix once
    cosine_sim = calculate_cosine_similarity_matrix(data)
    
    indices = pd.Series(data['title'])

    # Get movie recommendations
    recommended_movies = get_recommendations(movie_list, cosine_sim, indices, top_n)
    
    return recommended_movies
