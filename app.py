import streamlit as st
import pandas as pd
import numpy as np


st.title("Movie recommendation system")

#preprocessing data
movies = pd.read_csv('https://github.com/EnzeCa1/STAT542/raw/refs/heads/main/movies.dat', sep='::', names=['MovieID', 'Title', 'Genres'], engine='python', encoding='latin')
#similarity matrix was pre-generated for convenience, codes are in the html file
#S = pd.read_csv("similarity_matrix.csv")
#S.index = S.columns
movies['movie_id'] = 'm' + movies['MovieID'].astype(str)
movie_dict = pd.Series(movies['Title'].values, index=movies['movie_id']).to_dict()

#compute popularity score based on system 1
ratings = pd.read_csv('https://raw.githubusercontent.com/EnzeCa1/STAT542/refs/heads/main/ratings.dat', sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
ratings_matrix = ratings.pivot(index='UserID', columns='MovieID', values='Rating')

def compute_popularity_metrics(ratings_matrix, min_ratings=5):
    
    num_ratings = ratings_matrix.count(axis=0) 
    avg_rating = ratings_matrix.mean(axis=0, skipna=True)
    valid_movies = (num_ratings >= min_ratings).values # Filter movies with min_ratings threshold 

    # Population score is calculated by avg_rating weighted by num_ratings
    popularity_score = avg_rating * np.log1p(num_ratings)
    
    popularity_df = pd.DataFrame({
        'MovieID': ratings_matrix.columns,
        'NumRatings': num_ratings.values,
        'AvgRating': avg_rating.values,
        'PopularityScore': popularity_score.values
    })

    # Get top movies based on popularity score
    popularity_df = popularity_df[valid_movies].sort_values(by='PopularityScore', ascending=False)

    return popularity_df

#top_10_movies based on popularity score
popularity_df = compute_popularity_metrics(ratings_matrix, min_ratings=5)
top_10_movies = popularity_df.head(10)


movies['MovieID'] = movies['MovieID'].astype(int)
top_10_movies['MovieID'] = top_10_movies['MovieID'].astype(int)
top_10_movies = top_10_movies.merge(movies, on='MovieID', how='left')

S_100 = pd.read_csv('https://raw.githubusercontent.com/EnzeCa1/STAT542/refs/heads/main/S_100.csv')
#movie_ids = S.index
S_100.index = S_100.columns
sample_movie_ids = S_100.index #displaying 100 movies
#S_100 = S.loc[sample_movie_ids,sample_movie_ids]

#S_100.to_csv('S_100.csv',index = False)


movies_per_row = 4
num_movies = len(sample_movie_ids)
num_rows = (num_movies + movies_per_row - 1) // movies_per_row
ratings = {}

def myIBCF(newuser, S_100):
    movie_ids = S_100.index
    
    rated_mask = ~newuser.isna()
    unrated_mask = newuser.isna()
    
    predicted_ratings = pd.Series(index=movie_ids, dtype=float)
    
    for i in movie_ids[unrated_mask]:
        sim_row = S_100.loc[i]
        
        neighbors = sim_row.dropna().index.intersection(movie_ids[rated_mask])
        
        if len(neighbors) == 0:
            predicted_ratings[i] = np.nan
            continue
        
        numerator = (sim_row[neighbors] * newuser[neighbors]).sum()
        denominator = sim_row[neighbors].sum()
        
        if denominator == 0:
            predicted_ratings[i] = np.nan
        else:
            predicted_ratings[i] = numerator / denominator
    
    predicted_ratings[rated_mask] = np.nan
    
    sorted_predictions = predicted_ratings.dropna().sort_values(ascending=False)
    top_10 = sorted_predictions.head(10)
    
    if len(top_10) < 10:
        needed = 10 - len(top_10)
    
        recommended_ids = set(top_10['movie_id'].unique())
        rated_ids = set(newuser.dropna().index)

        fill_candidates = top_10_movies[
            (~top_10_movies['movie_id'].isin(recommended_ids)) &
            (~top_10_movies['movie_id'].isin(rated_ids))
        ]
        fill_movies = fill_candidates.head(needed)

        top_10 = pd.concat([top_10, fill_movies], ignore_index=True)
    
    return top_10

st.header("Rate These Movies")
st.write("Please provide ratings (1-5 stars) for the following movies. Rate as many as you can.")

for row_i in range(num_rows):
    cols = st.columns(movies_per_row)
    start_idx = row_i * movies_per_row
    end_idx = min(start_idx + movies_per_row, num_movies)

    for col, i in zip(cols, range(start_idx, end_idx)):
        mid = sample_movie_ids[i]
        title = movie_dict[mid]
        
        image_id = mid[1:]  
        image_path = f"https://liangfgithub.github.io/MovieImages/{image_id}.jpg"
        
        col.image(image_path, use_container_width=True)
        
        rating = col.selectbox(f"{title} ({mid})", 
                               options=["No Rating", "1", "2", "3", "4", "5"], 
                               index=0)
        ratings[mid] = np.nan if rating == "No Rating" else int(rating)

st.write("Once done, click below to get recommendations.")

if st.button("Get Recommendations"):
    newuser = pd.Series(np.nan, index=S_100.index)
    for mid, r in ratings.items():
        newuser[mid] = r

    top_10 = myIBCF(newuser, S_100)

    st.header("Top 10 Recommendations")

    top_10_df = top_10.to_frame(name='predicted_rating').reset_index()
    top_10_df.columns = ['movie_id', 'predicted_rating']
    top_10_df['movie_title'] = top_10_df['movie_id'].map(movie_dict)

    recs_per_row = 4
    num_recs = len(top_10_df)
    num_rows = (num_recs + recs_per_row - 1) // recs_per_row

    for row_i in range(num_rows):
        cols = st.columns(recs_per_row)
        start_idx = row_i * recs_per_row
        end_idx = min(start_idx + recs_per_row, num_recs)
    
        for col, idx in zip(cols, range(start_idx, end_idx)):
            rec_mid = top_10_df.at[idx, 'movie_id']
            rec_title = top_10_df.at[idx, 'movie_title']
            rec_rating = top_10_df.at[idx, 'predicted_rating']

            rec_image_id = rec_mid[1:]
            rec_image_path = f"https://liangfgithub.github.io/MovieImages/{rec_image_id}.jpg"
        
            col.image(rec_image_path, use_container_width=True)

            col.write(f"**{rec_title}** ({rec_mid})")
            col.write(f"Predicted Rating: {rec_rating:.2f}")
        
