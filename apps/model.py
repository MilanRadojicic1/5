import streamlit as st
import pandas as pd
import numpy as np


def app():


        st.title('RECOMMEND BY GENRE')

        st.write('This is the Movie Recommender by Genre!')
        st.write('Enter a Genre you feel like watching and we will output the highest rated movies for said Genre!')


        new_movieDF = pd.read_csv("./FYP_file.csv")
        new_movieDF['Genre'] = new_movieDF['Genre'].astype('str')
        new_movieDF['Genre'] = new_movieDF['Genre'].str.split()

        s = new_movieDF.apply(lambda x: pd.Series(x['Genre']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'genre'

        gen_md = new_movieDF.drop('Genre', axis=1).join(s)
        gen_md['genre'] = gen_md['genre'].str.split(",").str[0]
        gen_md = new_movieDF.explode('Genre')
        gen_md['Genre'] = gen_md['Genre'].str.replace(',', '')

        a = pd.unique(gen_md['Genre'])

        title1 = st.selectbox('Pick a movie genra', a, key=1)
        number1 = st.number_input('Enter the Number of recommended movies', min_value=0, max_value=30, step=1)


        if title1 and number1 > 0:

            def build_chart(genre, percentile=0.85):

                df = gen_md[gen_md['Genre'] == genre]
                vote_counts = df['Count of votes']
                vote_averages = df['Rating']

                C = vote_averages.mean()
                m = vote_counts.quantile(percentile)
                df = df.drop_duplicates(subset=['Title'])

                qualified = df[(df['Count of votes'] >= m)][
                    ['Title', 'Release_Year', 'Count of votes', 'Rating', 'Genre', 'Synopsis']]
                qualified['wr'] = qualified.apply(lambda x: (x['Count of votes'] / (x['Count of votes'] + m) * x['Rating']) + (
                            m / (m + x['Count of votes']) * C), axis=1)
                qualified = qualified.sort_values('wr', ascending=False)[0:number1]

                return qualified


            with st.spinner("Waiting for movie"):
                st.write(build_chart(title1))

