
import numpy as np
import pandas as pd

"""### Reading Files ###"""

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

#Reading 1st Entry in the file
print(movies.head(1))

#Reading 1st Entry in the file
print(credits.head(1))

"""### Data preprocessing ###"""

#merge both columns of the file
movies = movies.merge(credits, on= 'title')
print(movies.head(1))

#Selection of columns which has required info for recommendation
movies = movies[['movie_id','title','genres','overview','keywords','cast','crew']]
print(movies.head(1))

#checking for null values in the columns
print(movies.isnull().sum())

#dropping rows with null value
movies.dropna(inplace=True)

print(movies.iloc[0].genres)

#creating a function to store the value from name in thr columns
import ast
def convert(obj):
    l  = []
    for i  in ast.literal_eval(obj):
        l.append(i['name'])
    return l

#applying above function for the genere column
movies['genres'] = movies['genres'].apply(convert)
print(movies.head())

#applying above function for the keyword column
movies['keywords'] = movies['keywords'].apply(convert)
print(movies.head())

#function to only select first three names in the cast column
import ast
def convert3(obj):
    counter = 0
    l  = []
    for i  in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
            counter = counter+1
    return l

movies['cast'] =movies['cast'].apply(convert3)
print(movies.head())

#function to only have the director name in the crew
def fetch_direct(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director' :
            l.append(i['name'])
            break
    return l

movies['crew'] =movies['crew'].apply(fetch_direct)

#converting the the str to list of words
movies['overview'] = movies['overview'].apply(lambda x:x.split())

#removing spaces in the all the coolumns
movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

#combining all the 4 columns into one called tags
movies['tags'] = movies['overview']+movies['cast']+movies["crew"]+movies['keywords']

#new dataframe only woth 3 columns
new_df  = movies[['movie_id','title','tags']]
#joining and removing capital words in the column
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

print(new_df)

#importing sklearn to remove the the connecting english words and vectorising the data
#!pip install scikit-learn
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 5000, stop_words= 'english')

#tags column into array
vectors = cv.fit_transform(new_df['tags']).toarray()

#!pip install nltk

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

#cosine function to know the similarity between two vector data pts
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

#function to recommend
def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse= True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recommend('Batman')

"""### creation pkl file ###"""

import pickle
pickle.dump(new_df,open('movies.pkl','wb'))

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))