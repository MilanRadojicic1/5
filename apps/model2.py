import streamlit as st
import pandas as pd

def app():
        st.title('RECOMMEND BY DIRECTOR')

        st.write('This is the Movie Recommender by Director!')
        st.write('Enter a Director you love and we will output the highest rated movies the Director has made!')

        new_movieDF = pd.read_csv("./FYP_file.csv")

        new_movieDF['Genre'] = new_movieDF['Genre'].astype('str')

        new_movieDF['Genre'] = new_movieDF['Genre'].str.split()

        director_md = new_movieDF
        director_md['Director'] = director_md['Director'].str.replace(" ", "")
        director_md = director_md.explode('Director')

        a = pd.unique(director_md['Director'])
        title2 = st.selectbox('Movie director', a, key=2)
        number2 = st.number_input('Enter the Number of movies', min_value=0, max_value=30, step=1, key=2)

        if number2 > 0:

            def director_recommender(director, percentile=0.85):
                df = director_md[director_md['Director'] == director]
                vote_counts = df['Count of votes']
                vote_averages = df['Rating']

                C = vote_averages.mean()
                m = vote_counts.quantile(percentile)
                df = df.drop_duplicates(subset=['Title'])

                qualified = df[(df['Count of votes'] >= m)][
                    ['Title', 'Release_Year', 'Count of votes', 'Rating', 'Genre', 'Synopsis']]
                qualified['wr'] = qualified.apply(lambda x: (x['Count of votes'] / (x['Count of votes'] + m) * x['Rating']) + (m / (m + x['Count of votes']) * C), axis=1)
                qualified = qualified.sort_values('wr', ascending=False)[0:number2]

                return qualified



            with st.spinner("Waiting for movie"):
                st.write(director_recommender(title2))




