import streamlit as st
import pandas as pd

def app():


    st.title('RECOMMEND BY MOVIE DESCRIPTION AND METADATA')

    st.write('This is the Movie Recommender by Movie Description and Metadata!')
    st.write('Enter your favorite movie name and we will output movies that have a similar description,director,actor,title and genre!')


    one = pd.read_csv('one.csv')
    two = pd.read_csv('two.csv')
    three = pd.read_csv('three.csv')
    four = pd.read_csv('four.csv')
    new1 = one.append(two)
    new2 = three.append(new1)
    new_movieDF = four.append(new2)



    from ast import literal_eval
    new_movieDF['Genre'] = new_movieDF.Genre.apply(literal_eval)
    new_movieDF['lemmatized_text'] = new_movieDF.lemmatized_text.apply(literal_eval)

    filledna = new_movieDF.drop_duplicates(subset='Title')
    filledna = new_movieDF[new_movieDF['Rating'] >= 4.0]
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

    filledna.head(2)

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

    @st.cache(allow_output_mutation=True)
    def count_matrix(filledna):
        count_matrix = vectorizer.fit_transform(filledna)
        return (count_matrix)


    @st.cache(allow_output_mutation=True)
    def cosine_sim(count_matrix):
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        return (cosine_sim)

    count_matrix = count_matrix(filledna['soup'])

    cosine_sim2 = cosine_sim(count_matrix)


    filledna = filledna.reset_index()
    indices2 = pd.Series(filledna.index, index=filledna['Capital_Title'])

    a = pd.unique(filledna['Capital_Title'])


    title4 = st.selectbox('Pick a movie', a,key=4)
    number4 = st.number_input('Enter the Number of recommended movies',min_value=0, max_value=30,step=1)
    number4 = number4+1


    if number4>1:

        def description_metadata_recommendations(title, cosine_sim2=cosine_sim2):
            idx = indices2[title]

            # Get the pairwsie similarity scores of all movies with that movie
            sim_scores = list(enumerate(cosine_sim2[idx]))

            # Sort the movies based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the scores of the 10 most similar movies
            sim_scores = sim_scores[1:number4]

            # Get the movie indices
            movie_indices = [i[0] for i in sim_scores]

            # Return the top 10 most similar movies
            return filledna['Capital_Title'].iloc[movie_indices]




        with st.spinner("Waiting for movie"):
            st.write(description_metadata_recommendations(title4, cosine_sim2))

