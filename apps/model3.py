import streamlit as st
import pandas as pd

def app():
    st.title('RECOMMEND BY ACTOR')

    st.write('This is the Movie Recommender by Actor!')
    st.write('Enter a Actor you love and we will output the highest rated movies they have been in!')

    new_movieDF = pd.read_csv("./FYP_file.csv")

    new_movieDF['Genre'] = new_movieDF['Genre'].astype('str')

    new_movieDF['Genre'] = new_movieDF['Genre'].str.split()

    actor_md = new_movieDF
    actor_md = actor_md.dropna()
    actor_md['Actor'] = actor_md['Actor'].str.replace(' ', '')
    actor_md['Actor'] = actor_md['Actor'].str.split(',')
    actor_md = actor_md.explode('Actor')

    a = pd.unique(actor_md['Actor'])

    title3 = st.selectbox('Pick an actor', a,key=3)
    number3 = st.number_input('Enter the Number of recommended movies',min_value=0, max_value=30,step=1, key=3)


    def actor_recommender(actor, percentile=0.85):
        df = actor_md[actor_md['Actor'] == actor]
        vote_counts = df['Count of votes']
        vote_averages = df['Rating']

        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)
        df = df.drop_duplicates(subset=['Title'])

        qualified = df[(df['Count of votes'] >= m)][
            ['Title', 'Release_Year', 'Count of votes', 'Rating', 'Genre', 'Synopsis']]
        qualified['wr'] = qualified.apply(lambda x: (x['Count of votes'] / (x['Count of votes'] + m) * x['Rating']) + (m / (m + x['Count of votes']) * C), axis=1)
        qualified = qualified.sort_values('wr', ascending=False)[0:number3]

        return qualified






    if number3 > 0:
        with st.spinner("Waiting for movie"):
            st.write(actor_recommender(title3))





