# import graphlab
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from collections import Counter

def load_data(filename, colnames=None):
    if colnames:
        df = pd.read_csv(filename, sep = '\t',header = None, names=colnames)
    else:
        df = pd.read_csv(filename, sep = '\t')
    return df

def hist(df, column, x_lab, y_lab, title, bins=10):
    #takes in pandas dataframe, and column
    #prints out histogram of column
    plt.hist(df[column], bins=bins)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()

def scatter(df, x_lab, y_lab, title, bins=10):
    #takes in pandas dataframe, and column
    #prints out histogram of column
    plt.scatter(df[x_lab], df[y_lab])
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    colnames = ['user_id', 'item_id', 'rating']
    df = load_data('/Users/etownbetty/Documents/Galvanize/Recommender/recommender-case-study-files/data/ratings.dat')
    hist(df, 'rating', 'User Ratings', 'Frequency', 'User Rating Hist', bins=40)

    a = df.groupby('user_id') \
      .agg({'joke_id': len, 'rating': lambda x: x.mean()}) \
      .rename(columns={'joke_id':'num_jokes_rated','rating':'avg_rating'})
    hist(a, 'avg_rating', 'Avg User Ratings', 'Frequency', 'Avg User Rating Hist', bins=40)
    hist(a, 'num_jokes_rated', 'Total Jokes Rated', 'Frequency', 'User Num Jokes', bins=40)

    print("Mean number of jokes rated: {}".format(a['num_jokes_rated'].mean())
    print("Mean joke rating per user: {}".format(a['avg_rating'].mean()))
    joke_count = Counter(a['num_jokes_rated'])
    joke_count.most_common(1)
    print("Mean joke rating per user, +100 jokes: {}".format(a[a['num_jokes_rated']==100]['avg_rating'].mean()))
    scatter(a, 'num_jokes_rated', 'avg_rating', 'Avg User Rating by #Jokes Rated')

    jokes = df.groupby('joke_id') \
      .agg({'joke_id': len, 'rating': lambda x: x.mean()}) \
      .rename(columns={'joke_id':'num_joke_ratings','rating':'avg_rating'})
    jokes.sort('avg_rating', ascending=False).head()
    jokes.sort('avg_rating', ascending=True).head()
    scatter(jokes, 'num_joke_ratings', 'avg_rating', 'Number of Jokes Rated by Avg Rating')
