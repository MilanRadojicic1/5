import streamlit as st
import pandas as pd
from ast import literal_eval


def app():
    st.title('RECOMMEND BY MOVIE DESCRIPTION AND METADATA BUT PRIORITIZE A MOVIES RATING AND NUMBER OF VOTES ')

    st.write('This is the Movie Recommender by Movie Description,Metadata, Movie Rating and Number of Votes!')
    st.write('Enter your favorite movie name and we will output movies that have a similar description,director,actor,title and genre but we will prioritize the movies rating and number of votes!')

    one = pd.read_csv('one.csv')
    two = pd.read_csv('two.csv')
    three = pd.read_csv('three.csv')
    four = pd.read_csv('four.csv')

    new1 = one.append(two)
    new2 = three.append(new1)
    new_movieDF = four.append(new2)


    new_movieDF['Genre'] = new_movieDF.Genre.apply(literal_eval)
    new_movieDF['lemmatized_text'] = new_movieDF.lemmatized_text.apply(literal_eval)


    filledna = new_movieDF.drop_duplicates(subset='Title')

    filledna = new_movieDF[new_movieDF['Rating'] >= 7.0]
    filledna = new_movieDF[new_movieDF['Count of votes'] >= 5000]

    filledna = filledna.fillna('')

    filledna['Genre'] = [','.join(map(str, l)) for l in filledna['Genre']]
    filledna['Genre'] = filledna['Genre'].str.replace(',,', ',')

    filledna['liststring'] = [','.join(map(str, l)) for l in filledna['lemmatized_text']]
    a = filledna

    def clean_data(x):
        return str.lower(x)

    filledna['Capital_Title'] = filledna['Title']

    features = ['Title', 'Director', 'Actor', 'Genre', 'liststring']
    filledna = filledna[features]

    for feature in features:
        filledna[feature] = filledna[feature].apply(clean_data)


    def create_soup(x):
        return x['Title'] + ' ' + x['Director'] + ' ' + x['Actor'] + ' ' + x['Genre'] + ' ' + x['liststring']


    filledna['soup'] = filledna.apply(create_soup, axis=1)

    filledna['Capital_Title'] = a['Title']
    filledna['Count of votes'] = a['Count of votes']
    filledna['Rating'] = a['Rating']
    filledna['Release_Year'] = a['Release_Year']

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    vectorizer = TfidfVectorizer()

    vectorizer = TfidfVectorizer()

    @st.cache(allow_output_mutation=True)
    def count_matrix(metadata_recomendation):
        count_matrix = vectorizer.fit_transform(metadata_recomendation)
        return (count_matrix)


    @st.cache(allow_output_mutation=True)
    def cosine_sim(count_matrix):
        cosine_sim5 = cosine_similarity(count_matrix, count_matrix)
        return (cosine_sim5)



    count_matrix = count_matrix(filledna['soup'])

    cosine_sim5 = cosine_sim(count_matrix)

    filledna = filledna.reset_index()
    indices2 = pd.Series(filledna.index, index=filledna['Capital_Title'])

    vote_counts = new_movieDF[new_movieDF['Count of votes'].notnull()]['Count of votes'].astype('int')
    vote_averages = new_movieDF[new_movieDF['Rating'].notnull()]['Rating'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)

    def weighted_rating(x):
        v = x['Count of votes']
        R = x['Rating']
        return (v / (v + m) * R) + (m / (m + v) * C)


    a = pd.unique(filledna['Capital_Title'])


    title7 = st.selectbox('Pick a movie', a,key=7)
    number7 = st.number_input('Enter the Number of recommended movies',min_value=0, max_value=30,step=1, key=7)



    def improved_recommendations(title):
        idx = indices2[title]
        sim_scores = list(enumerate(cosine_sim5[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:30]
        movie_indices = [i[0] for i in sim_scores]

        movies = filledna.iloc[movie_indices][
            ['Capital_Title', 'Count of votes', 'Rating', 'Release_Year', 'Director', 'Actor']]
        vote_counts = movies[movies['Count of votes'].notnull()]['Count of votes'].astype('int')
        vote_averages = movies[movies['Rating'].notnull()]['Rating'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)
        qualified = movies[
            (movies['Count of votes'] >= m) & (movies['Count of votes'].notnull()) & (movies['Rating'].notnull())]
        qualified['Count of votes'] = qualified['Count of votes'].astype('int')
        qualified['Rating'] = qualified['Rating'].astype('int')
        qualified['wr'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('wr', ascending=False)[0:number7]
        return qualified




    if number7>0:
        with st.spinner("Waiting for movie"):
            st.write(improved_recommendations(title7))



