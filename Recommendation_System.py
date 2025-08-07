#!/usr/bin/env python
# coding: utf-8

# # the Recommendation System

# In[1]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge data
movie_data = pd.merge(ratings, movies, on='movieId')

# Create user-item matrix
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaNs with 0 for similarity calculation
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# Compute similarity matrix
similarity_matrix = cosine_similarity(user_movie_matrix_filled)

# Recommend movies for a given user
def recommend_movies(user_id, num_recommendations=5):
    user_index = user_id - 1  # since userId starts at 1
    similarity_scores = list(enumerate(similarity_matrix[user_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]

    similar_users = [i[0] for i in similarity_scores[:10]]
    recommended_movies = user_movie_matrix.iloc[similar_users].mean().sort_values(ascending=False)

    user_watched = user_movie_matrix.iloc[user_index][user_movie_matrix.iloc[user_index].notna()].index
    recommendations = recommended_movies.drop(labels=user_watched).head(num_recommendations)
    return recommendations.index.tolist()

# Example usage
print("Recommended movies for User 1:")
print(recommend_movies(user_id=1))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




