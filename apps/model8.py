import pandas as pd
import streamlit as st

def app():


    st.title('RECOMMEND MOVIES BY USER BASED SIMILARITIES!')

    st.write('This is the Movie Recommender which outputs movies by similarity in individuals ratings of movies!')
    st.write('Enter a movie name and rate it and we will output movies from other people that have rated the same movies similar to you!')

    movie = pd.read_csv('small_movies.csv')
    rating = pd.read_csv('small_ratings.csv')

    movie['year'] = movie.title.str.extract('(\\d\d\d\d\))', expand=False)
    # Removing the parentheses
    movie['year'] = movie.year.str.extract('(\d\d\d\d)', expand=False)
    # Removing the years from the 'title' column
    movie['title'] = movie.title.str.replace('(\(\d\d\d\d\))', '')
    # Applying the strip function to get rid of any ending whitespace characters that may have appeared
    movie['title'] = movie['title'].apply(lambda x: x.strip())



    movie.drop(columns=['genres'], inplace=True)

    rating.drop(columns=['timestamp'],inplace=True)


    n = 5

    titles = []
    ratings = []

    a = pd.unique(movie['title'])

    one_title_name = st.selectbox("Pick a movie (1):",a,key="10")
    one_title_rating = st.number_input('Enter the rating of selected movie',min_value=0, max_value=5,step=1,key="10")
    titles.append(one_title_name)
    ratings.append(one_title_rating)

    two_title_name = st.selectbox("Pick a movie (2):",a,key="11")
    two_title_rating = st.number_input('Enter the rating of selected movie',min_value=0, max_value=5,step=1,key="11")
    titles.append(two_title_name)
    ratings.append(two_title_rating)

    three_title_name = st.selectbox("Pick a movie (3):",a,key="12")
    three_title_rating = st.number_input('Enter the rating of selected movie',min_value=0, max_value=5,step=1,key="12")
    titles.append(three_title_name)
    ratings.append(three_title_rating)

    four_title_name = st.selectbox("Pick a movie (4):",a,key="13")
    four_title_rating = st.number_input('Enter the rating of selected movie',min_value=0, max_value=5,step=1,key="13")
    titles.append(four_title_name)
    ratings.append(four_title_rating)

    five_title_name = st.selectbox("Pick a movie (5):",a,key="14")
    five_title_rating = st.number_input('Enter the rating of selected movie',min_value=0, max_value=5,step=1,key="14")
    titles.append(five_title_name)
    ratings.append(five_title_rating)

    if  one_title_rating>0 and  two_title_rating>0  and three_title_rating>0 and four_title_rating>0  and five_title_rating>0:

        b = {'title': titles, 'rating': ratings}
        inputMovie = pd.DataFrame(b)

        inputMovie['rating'] = inputMovie['rating'].astype(float)

        #Filtering out the movies by title
        Id = movie[movie['title'].isin(inputMovie['title'].tolist())]
        #Then merging it so we can get the movieId. It's implicitly merging it by title.
        inputMovie = pd.merge(Id, inputMovie)
        #Dropping information we won't use from the input dataframe
        inputMovie = inputMovie.drop('year', 1)

        # Filtering out users that have watched movies that the input has watched and storing it
        users = rating[rating['movieId'].isin(inputMovie['movieId'].tolist())]

        # Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
        userSubsetGroup = users.groupby(['userId'])

        #Sorting it so users with movie most in common with the input will have priority
        userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)


        from math import sqrt

        # Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
        pearsonCorDict = {}

        # For every user group in our subset
        for name, group in userSubsetGroup:
            # Let's start by sorting the input and current user group so the values aren't mixed up later on
            group = group.sort_values(by='movieId')
            inputMovie = inputMovie.sort_values(by='movieId')
            # Get the N for the formula
            n = len(group)
            # Get the review scores for the movies that they both have in common
            temp = inputMovie[inputMovie['movieId'].isin(group['movieId'].tolist())]
            # And then store them in a temporary buffer variable in a list format to facilitate future calculations
            tempRatingList = temp['rating'].tolist()
            # put the current user group reviews in a list format
            tempGroupList = group['rating'].tolist()
            # Now let's calculate the pearson correlation between two users, so called, x and y
            Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(n)
            Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(n)
            Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(
                tempGroupList) / float(n)

            # If the denominator is different than zero, then divide, else, 0 correlation.
            if Sxx != 0 and Syy != 0:
                pearsonCorDict[name] = Sxy / sqrt(Sxx * Syy)
            else:
                pearsonCorDict[name] = 0


        pearsonDF = pd.DataFrame.from_dict(pearsonCorDict, orient='index')
        pearsonDF.columns = ['similarityIndex']
        pearsonDF['userId'] = pearsonDF.index
        pearsonDF.index = range(len(pearsonDF))


        topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]


        topUsersRating=topUsers.merge(rating, left_on='userId', right_on='userId', how='inner')


        topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']


        #Applies a sum to the topUsers after grouping it up by userId
        tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
        tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']

        # Creates an empty dataframe
        recommendation_df = pd.DataFrame()
        # Now we take the weighted average
        recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / \
                                                                     tempTopUsersRating['sum_similarityIndex']
        recommendation_df['movieId'] = tempTopUsersRating.index

        recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)



        with st.spinner("Waiting for movie"):
            (st.write(movie.loc[movie['movieId'].isin(recommendation_df.head(5)['movieId'].tolist())]))