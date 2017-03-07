import graphlab
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from helper_hao import get_ratings

def recom_fac_model(sdf):
    '''This is the model using factorization_recommender
        returns a recommender model'''
    recommender = graphlab.recommender.factorization_recommender.create(sdf, user_id='user_id', item_id='joke_id',target = 'rating',solver='als', num_factors=20)
    return recommender

def recom_rank_fac_modle(df):
    '''This is the model using ranking_factorization_recommender
        returns a recommender model'''
    recommender = graphlab.recommender.ranking_factorization_recommender.create(sdf, user_id='user_id', item_id='joke_id',target = 'rating',solver='sgd', num_factors=20)
    return recommender

def user_joke_info(df):
    '''returns a tuple of the ratio of unrated jokes, number of users, and number of jokes'''
    unrated = df.shape[0]
    users = len(df['user_id'].unique())
    jokes = len(df['joke_id'].unique())
    return unrated*1.0/(users*jokes), users, jokes

def recommend(model, K):
    '''return the dataframe for recommending the k highest scored items for each user'''
    recom = model.recommend(k=K)
    return recom.to_dataframe()

def get_topn(item_n = 100, top_percent=5):
    return int(item_n*top_percent)

def write_csv(df, filepath):
    '''Write dataframe into a csv file in provided filepath'''
    try:
        df.to_csv(filepath)
    except:
        print('Error in write csv file!')
    return

if __name__ == "__main__":
    sample_sub_fname = "/Users/haowei/Documents/gal/Day_51_recommender_case_study/data/sample_submission.csv"
    datafile = "/Users/haowei/Documents/gal/Day_51_recommender_case_study/data/ratings.dat"
    output_file1 = "/Users/haowei/Documents/gal/Day_51_recommender_case_study/data/recommend_fac_model.csv"
    output_file2 = "/Users/haowei/Documents/gal/Day_51_recommender_case_study/data/recommend_rank_model.csv"

    df, sdf = get_ratings()
    model1 = recom_fac_model(sdf)
    modle2 = recom_rank_fac_modle(sdf)

    rate = 0.05
    unrated_percent, users, jokes = user_joke_info()
    K = get_topn(jokes,rate)
    recommender1 = recommend(model1, K)
    recommender2 = recommend(model2, K)

    write_csv(recommender1, output_file1)
    write_csv(recommender2, output_file2)
