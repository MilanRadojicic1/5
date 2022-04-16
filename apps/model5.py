import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ast import literal_eval


def app():


    st.title('RECOMMEND BY MOVIE DESCRIPTION')

    st.write('This is the Movie Recommender by Movie Description!')
    st.write('Enter your favorite movie name and we will output movies that have a similar description!')


    one = pd.read_csv('one.csv')
    two = pd.read_csv('two.csv')
    three = pd.read_csv('three.csv')
    four = pd.read_csv('four.csv')

    new1 = one.append(two)
    new2 = three.append(new1)
    new_movieDF = four.append(new2)

    new_movieDF['Genre'] = new_movieDF.Genre.apply(literal_eval)
    new_movieDF['lemmatized_text'] = new_movieDF.lemmatized_text.apply(literal_eval)




    lemmatized_text_description = new_movieDF[new_movieDF['Rating'] >= 4.0]
    lemmatized_text_description = new_movieDF[new_movieDF['Count of votes'] >= 5000]

    lemmatized_text_description = lemmatized_text_description.fillna('')

    lemmatized_text_description['lemmeatized_string_movie_description'] = lemmatized_text_description['lemmatized_text'].apply(lambda x: ','.join(map(str, x)))



    vectorizer = TfidfVectorizer()


    @st.cache(allow_output_mutation=True)
    def count_matrix(filledna):
        count_matrix = vectorizer.fit_transform(filledna)
        return (count_matrix)


    @st.cache(allow_output_mutation=True)
    def cosine_sim(count_matrix):
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        return (cosine_sim)

    count_matrix = count_matrix(lemmatized_text_description['lemmeatized_string_movie_description'])

    cosine_sim = cosine_sim(count_matrix)

    lemmatized_text_description = lemmatized_text_description.reset_index()
    indices = pd.Series(lemmatized_text_description.index, index=lemmatized_text_description['Title'])


    a = pd.unique(lemmatized_text_description['Title'])

    title5 = st.selectbox('Pick a movie', a,key=5)
    number5 = st.number_input('Enter the Number of recommended movies',min_value=0, max_value=30,step=1)
    number5 = number5+1

    def description_get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:number5]
        movie_indices = [i[0] for i in sim_scores]
        return lemmatized_text_description['Title'].iloc[movie_indices]




    if number5>1:
        with st.spinner("Waiting for movie"):
            st.write(description_get_recommendations(title5, cosine_sim))



