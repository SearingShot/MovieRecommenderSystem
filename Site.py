import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np

# Importing The Datasets 
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')

## Columns that we are going to Consider
# genres
# id
# keywords
# title
# overview
# cast
# crew

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.dropna(inplace=True)  # Dropping the missing data if found any.

# We want to re-arrange irregular columns like genres, keywords, cast and crew, 
# because they are in 'string of list of dictionaries' form.

import json


def convert(obj):
    # parsing json string into list of dictionaries 
    obj = json.loads(obj)
    list1 = [i['name'] for i in obj]
    return list1


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


# For Cast Column, we will take only 5 cast
def convert2(obj):
    # parsing json string into list of dictionaries 
    obj = json.loads(obj)
    Counter = 0
    list1 = []
    for i in obj:
        if Counter != 5:
            list1.append(i['name'])
            Counter += 1
        else:
            break
    return list1


movies['cast'] = movies['cast'].apply(convert2)


# For Crew Column, we will take only directors
def fetch_director(obj):
    obj = json.loads(obj)
    list1 = []
    for i in obj:
        if i['job'] == 'Director':
            list1.append(i['name'])
            break
    return list1


movies['crew'] = movies['crew'].apply(fetch_director)

# Since Overview is in string form, so Converting Overview Column into list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Now we'll remove Whitespaces from our Columns
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Now we will create a new column Named Tag and concatenate other columns like overview, genres, keywords, cast, crew in it.
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Now We will create a new dataframe, which will only consists of these columns - movie_id, title, and Tags
New_df = movies[['movie_id', 'title', 'tags']]

New_df['tags'] = New_df['tags'].apply(lambda x: " ".join(x))  # Now converting Tags Column into strings

New_df['tags'] = New_df['tags'].apply(lambda x: x.lower())  # changing strings into lowercase strings

# Since We Dont Want Repition of words like Dance, Dancing, Danced, We will use nltk library to get their stem word dance.
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


New_df['tags'] = New_df['tags'].apply(stem)

# Converting Tags Into Vector form
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(New_df['tags']).toarray()

cv.get_feature_names_out()

# Now we'll be calculating distance of these movie vectors from each other,
# because lesser the distance between two vectors would mean , the similarity between movies are more and vice versa
# This Distance will be calculated through cosine functions

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)


# Now We'll make a recommend function which will provide us with similar movies

def recommend(movies):
    movie_index = New_df[New_df['title'] == movies].index[0]  # Fetching index of the movies
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    # here we sort the movies in similarity manner by sorting the array in reverse order as higher the score,
    # higher the similarity. while also maintaining their index using enumerate function
    # and lambda is used so that sorting function uses value as basis for sorting and not the index.

    for i in movies_list:
        print(New_df.iloc[i[0]].title)  # Fetching the movies by titles


recommend('The Avengers')


def fetch_poster(movie_id):
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(
            movie_id))
    data = response.json()
    poster = 'https://image.tmdb.org/t/p/w500/' + data['poster_path']
    return poster


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))  # fetching the movie poster from api
    return recommended_movies, recommended_movies_posters


movies_dict = New_df.to_dict()
movies = pd.DataFrame(movies_dict)

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Select Your Movie',
    movies['title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])