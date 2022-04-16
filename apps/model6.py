import streamlit as st
import pandas as pd
from ast import literal_eval


def app():
    st.title('RECOMMEND BY MOVIE METADATA')

    st.write('This is the Movie Recommender by Movie METADATA!')
    st.write('Enter your favorite movie name and we will output movies that have a similar director,actor,title and genre!')

    one = pd.read_csv('one.csv')
    two = pd.read_csv('two.csv')
    three = pd.read_csv('three.csv')
    four = pd.read_csv('four.csv')

    new1 = one.append(two)
    new2 = three.append(new1)
    new_movieDF = four.append(new2)


    new_movieDF['Genre'] = new_movieDF.Genre.apply(literal_eval)
    new_movieDF['lemmatized_text'] = new_movieDF.lemmatized_text.apply(literal_eval)

    metadata_recomendation = new_movieDF[new_movieDF['Rating'] >= 6.0]
    metadata_recomendation = metadata_recomendation.fillna('')

    metadata_recomendation['Genre'] = [','.join(map(str, l)) for l in metadata_recomendation['Genre']]
    metadata_recomendation['Genre'] = metadata_recomendation['Genre'].str.replace(',,', ',')

    def clean_data(x):
        return str.lower(x)

    metadata_recomendation['Capital_Title'] = metadata_recomendation['Title']
    a = metadata_recomendation

    features = ['Title', 'Director', 'Actor', 'Genre']
    metadata_recomendation = metadata_recomendation[features]

    for feature in features:
        metadata_recomendation[feature] = metadata_recomendation[feature].apply(clean_data)

    def create_soup(x):
        return x['Title'] + ' ' + x['Director'] + ' ' + x['Actor'] + ' ' + x['Genre']

    metadata_recomendation['soup'] = metadata_recomendation.apply(create_soup, axis=1)

    metadata_recomendation['Capital_Title'] = a['Capital_Title']

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    vectorizer = TfidfVectorizer()

    @st.cache(allow_output_mutation=True)
    def count_matrix(metadata_recomendation):
        count_matrix = vectorizer.fit_transform(metadata_recomendation)
        return (count_matrix)


    @st.cache(allow_output_mutation=True)
    def cosine_sim(count_matrix):
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        return (cosine_sim)

    count_matrix = count_matrix(metadata_recomendation['soup'])

    cosine_sim3 = cosine_sim(count_matrix)


    metadata_recomendation = metadata_recomendation.reset_index()
    indices = pd.Series(metadata_recomendation.index, index=metadata_recomendation['Capital_Title'])

    c = pd.unique(metadata_recomendation['Capital_Title'])

    title6 = st.selectbox('Pick a movie', c,key=6)
    number6 = st.number_input('Enter the Number of recommended movies',min_value=0, max_value=30,step=1)
    number6 = number6+1

    def description_metadata_recommendations(title, cosine_sim3=cosine_sim3):
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim3[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:number6]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return metadata_recomendation['Capital_Title'].iloc[movie_indices]


    if number6>1:
        with st.spinner("Waiting for movie"):
            st.write(description_metadata_recommendations(title6, cosine_sim3))









