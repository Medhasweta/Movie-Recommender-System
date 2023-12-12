# Movie Recommender System README

## Overview
This Movie Recommender System is a Python-based project that analyzes movie data to recommend similar movies based on user preferences. It leverages Pandas for data manipulation, NumPy for numerical operations, and other libraries for text processing and similarity calculations.

## Requirements
- Python 3.x
- Libraries: NumPy, Pandas, ast, sklearn, nltk, pickle, streamlit, requests

## Dataset
The dataset used can be found on Kaggle: TMDB Movie Metadata. It includes two files:  

- tmdb_5000_movies.csv  
- tmdb_5000_credits.csv  

## Usage  
- Load and preprocess the data from the provided CSV files.  
- Conduct data cleaning, including handling null values and merging datasets.  
- Extract and transform relevant information from the dataset for the recommendation process.  
- Implement text processing and vectorization for feature extraction.  
- Use cosine similarity for calculating similarities between movies.  
- Implement a function recommend(movie) to get movie recommendations.  
- Create and utilize a Streamlit application for an interactive recommendation system.  
- Run the Streamlit app.

## Features
- Movie data processing and analysis.
- Text vectorization for feature extraction.
- Cosine similarity for recommendation.
- Streamlit application for an interactive recommendation system.
  
