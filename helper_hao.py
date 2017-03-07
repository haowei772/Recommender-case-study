import graphlab
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def get_ratings():
    '''
    Returns a tuple containing:
        - a dataframe of ratings
        - a sparse matrix where rows correspond to users and columns correspond
        to jokess. Each element is the user's rating for that movie.
    '''
    filename = '/Users/haowei/Documents/gal/Day_51_recommender_case_study/data/ratings.dat'
    df = pd.read_csv(filename)
    return df, graphlab.SFrame(df)
