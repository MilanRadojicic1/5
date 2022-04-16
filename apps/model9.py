from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st


def app():
    st.title('RECOMMEND MOVIES BY ITEM BASED COLLABORATIVE SIMILARITIES!')

    st.write('This is the Movie Recommender which outputs movies by similarity between items calculated using the ratings users have given to items!')
    st.write('Enter a movie name and we will output movies from other people that enjoyed the movie you have entered!')

    movie = pd.read_csv('small_movies.csv')
    rating = pd.read_csv('small_ratings.csv')

    movie['year'] = movie.title.str.extract('(\\d\d\d\d\))', expand=False)
    # Removing the parentheses
    movie['year'] = movie.year.str.extract('(\d\d\d\d)', expand=False)
    # Removing the years from the 'title' column
    movie['title'] = movie.title.str.replace('(\(\d\d\d\d\))', '')
    # Applying the strip function to get rid of any ending whitespace characters that may have appeared
    movie['title'] = movie['title'].apply(lambda x: x.strip())


    a = pd.unique(movie['title'])

    title20 = st.selectbox('Pick a movie', a,key=20)
    number20 = st.number_input('Enter the Number of recommended movies',min_value=0, max_value=30,step=1)

    if title20 and number20 > 0:




        final_dataset = rating.pivot_table(index='movieId', columns='userId', values='rating')


        final_dataset.fillna(0,inplace=True)

        csr_data = csr_matrix(final_dataset.values)
        final_dataset.reset_index(inplace=True)

        knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        knn.fit(csr_data)

        def get_movie_recommendation(movie_name):
            n_movies_to_reccomend = number20
            movie_list = movie[movie['title'].str.contains(movie_name)]
            if len(movie_list):
                movie_idx = movie_list.iloc[0]['movieId']  # movieId
                movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]  # userId acc to movieId
                distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend + 1)
                rec_movie_indices = sorted(list(zip(indices.squeeze(), distances.squeeze())), key=lambda x: x[1])[1::1]
                recommend_frame = []
                for val in rec_movie_indices:
                    movie_idx = final_dataset.iloc[val[0]]['movieId']
                    idx = movie[movie['movieId'] == movie_idx].index
                    recommend_frame.append(movie.iloc[idx]['title'].values[0])
                df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend + 1))
                return df
            else:
                return "No movies found. Please check your input"



        with st.spinner("Waiting for movie"):
            st.write(get_movie_recommendation(title20))


