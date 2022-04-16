
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import tweepy
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
nltk.download('stopwords')
import streamlit as st
from ast import literal_eval
nltk.download('punkt')


def app():


    st.title('RECOMMEND MOVIES BY USING INDIVIDUALS TWEETS,MENTIONS AND FRIENDS')

    st.write('This is the Movie Recommender which outputs movies depending on an individuals tweet content,mentions and people they follow!')
    st.write('Enter an individuals twitter handle and we will output movies depending on the entered individuals tweets,mentions and friend list!')


    st.write("Enter twitter name",)
    user_name = st.text_input('',key="31")
    number_of_tweets = st.number_input('Enter the Number of tweets you wish to pull in',min_value=100, max_value=500,step=20,key=40)
    number_of_friends = st.number_input('Enter the Number of friends ',min_value=10, max_value=50,step=5,key=50)
    number50 = st.number_input('Enter the Number of recommended movies',min_value=0, max_value=30,step=1,key=100)

    if user_name and number50>0:




        one = pd.read_csv('one.csv')
        two = pd.read_csv('two.csv')
        three = pd.read_csv('three.csv')
        four = pd.read_csv('four.csv')

        new1 = one.append(two)
        new2 = three.append(new1)
        new_movieDF = four.append(new2)

        new_movieDF = new_movieDF[new_movieDF['Rating'] >= 4.0]
        new_movieDF = new_movieDF[new_movieDF['Count of votes'] >= 1000]


        new_movieDF['Genre'] = new_movieDF.Genre.apply(literal_eval)
        new_movieDF['lemmatized_text'] = new_movieDF.lemmatized_text.apply(literal_eval)



        # Twitter connection credentials
        api_key = 'fBgubv7aNOfb9ndVW2PT5UEXO'
        api_secret_key = '14HUsseqPtSqIDQce8F1NKnfQmuS4CfQs2zWi0MALyyd7CVCoB'
        access_token = '1447495934384951297-yOo3pw61SjcHvfqlLlEyIcoLwgLqgh'
        access_token_secret = 'g0c7jsUdvF67VxkAs8vz29fNq2qpgnZspCufydLm9s6yX'

        # authenticate to the Twitter api/Tweepy
        auth = tweepy.OAuthHandler(api_key, api_secret_key)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)


        # INPUT: [string] screen_name
        # OUTPUT: [dataframe] account_df
        # This function takes in a screen_name and then returns a dataframe containing data about that particular account. It does
        # this by using the 'get_user' API method from Tweepy, and it searches for the relevant account by using the screen_name
        def create_account_df(account_name):
            account = api.get_user(screen_name=account_name)

            created_at = []
            screen_name = []
            friends_count = []
            followers_count = []
            favourites_count = []
            statuses_count = []

            created_at.append(account.created_at)
            screen_name.append(account.screen_name)
            friends_count.append(account.friends_count)
            followers_count.append(account.followers_count)
            favourites_count.append(account.favourites_count)
            statuses_count.append(account.statuses_count)

            account_df = pd.DataFrame({'created_at': created_at, 'screen_name': screen_name, 'friends_count': friends_count,
                                       'followers_count': followers_count, 'favourites_count': favourites_count,
                                       'statuses_count': statuses_count})

            return (account_df)


        def create_friends_df(account_name, number_of_friends):
            created_at = []
            name = []
            screen_name = []
            followers_count = []
            friends_count = []

            for follower in tweepy.Cursor(api.get_friends, screen_name=account_name, tweet_mode="extended").items(
                    number_of_friends):
                created_at.append(follower.created_at)
                name.append(follower.name)
                screen_name.append(follower.screen_name)
                followers_count.append(follower.followers_count)
                friends_count.append(follower.friends_count)

            friends_df = pd.DataFrame(
                {'created_at': created_at, 'name': name, 'screen_name': screen_name, 'followers_count': followers_count,
                 'friends_count': friends_count})

            friends_df['account'] = account_name

            return (friends_df)


        # INPUT: [string] account_name, [integer] number_of_tweets, [boolean] include_retweets
        # OUTPUT: [dataframe] timeline_df
        # This function takes in an screen_name and then returns a dataframe containing data about the tweets made by that account.
        # The number of tweets contained in the returned dataframe is dependent on the number_of_tweets variable that this function takes as input.
        # This function also takes a boolean value as an input parameter. Depending on this boolean value, the function will either
        # return a dataframe of tweets that include retweets or it will return a dataframe of tweets that exclude retweets
        def get_user_timeline(account_name, number_of_tweets, include_retweets):
            created_at = []
            tweet = []
            favorite_count = []
            retweet_count = []
            source = []
            is_quote_status = []
            favorited = []
            mentions = []
            hashtags = []

            for status in tweepy.Cursor(api.user_timeline, screen_name=account_name, include_rts=include_retweets,
                                        tweet_mode='extended').items(number_of_tweets):
                created_at.append(status.created_at)
                tweet.append(status.full_text)
                favorite_count.append(status.favorite_count)
                retweet_count.append(status.retweet_count)
                is_quote_status.append(status.is_quote_status)
                favorited.append(status.favorited)
                mention_list = []
                mentions.append([mention['screen_name'] for mention in status.entities['user_mentions'] if
                                 status.entities['user_mentions']])
                hashtags.append([hashtag['text'] for hashtag in status.entities['hashtags'] if status.entities['hashtags']])

            timeline_df = pd.DataFrame(
                {'created_at': created_at, 'tweet': tweet, 'favorite_count': favorite_count, 'retweet_count': retweet_count,
                 'is_quote_status': is_quote_status, 'favorited': favorited, 'mentions': mentions, 'hashtags': hashtags})

            # add an 'author' column to the dataframe that shows the screen_name of the account that made the tweet
            timeline_df['author'] = account_name

            return timeline_df




        # INPUT: [string] account_name, [integer] number_of_tweets
        # OUTPUT: [dataframe] timeline_df, [dataframe] timeline_df
        # This function takes in an account_name and returns 2 dataframes containing data about all the tweets posted by the account
        # under that account name. One dataframe will contain data about all tweets, including retweets. The other dataframe will
        # consist of original tweets only, and will exclude retweets. Both are necessary for visuals.
        def create_timeline_df(account_name, number_of_tweets):
            return get_user_timeline(account_name, number_of_tweets, include_retweets=True), get_user_timeline(account_name, number_of_tweets, include_retweets=False)


        # INPUT: [string] account_name, [integer] number_of_followers
        # OUTPUT: [dataframe] followers_df
        # This function takes in an account_name and returns a dataframe containing data about the accounts that follow that account.
        # This function also utilises the genderize function to assign a gender to the accounts, on the basis of their first_name.
        def create_followers_df(account_name, number_of_followers):
            created_at = []
            id = []
            name = []
            screen_name = []
            followers_count = []
            friends_count = []

            for follower in tweepy.Cursor(api.get_followers, screen_name=account_name, tweet_mode="extended").items(
                    number_of_followers):
                created_at.append(follower.created_at)
                id.append(follower.id)
                name.append(follower.name)
                screen_name.append(follower.screen_name)
                followers_count.append(follower.followers_count)
                friends_count.append(follower.friends_count)

            followers_df = pd.DataFrame({'created_at': created_at, 'id': id, 'name': name, 'screen_name': screen_name,
                                         'followers_count': followers_count, 'friends_count': friends_count})

            # add an 'author' column to the dataframe that shows the screen_name of the account that made the tweet
            followers_df['account'] = account_name

            return (followers_df)

        import time

        time.sleep(15)

        # run 'create_account_df' function for each account_name in order to fetch data for that specific account
        KJ_df = create_account_df(account_name=user_name)

        # join together the dataframes showing data about each account
        account_df = pd.concat([KJ_df], ignore_index=True)



        # run 'create_timeline_df' function for each account_name in order to fetch data for that specific account
        KJ_timeline_df_RT, KJ_timeline_df_no_RT = create_timeline_df(account_name=user_name, number_of_tweets = number_of_tweets)

        # join together the dataframes showing data about each account's timeline
        timeline_df_no_RT = pd.concat([KJ_timeline_df_no_RT], ignore_index=True)
        timeline_df_RT = pd.concat([KJ_timeline_df_RT], ignore_index=True)



        KJ_followers_df = create_followers_df(account_name=user_name, number_of_followers=20)

        # join together the dataframes showing data about each account's followers
        followers_df = pd.concat([KJ_followers_df], ignore_index=True)



        # run 'create_friends_df' function for each account_name in order to fetch data for that specific account
        KJ_friends_df = create_friends_df(account_name=user_name, number_of_friends = number_of_friends)


        # join together the dataframes showing data about each account's timeline
        friends_df = pd.concat([KJ_friends_df], ignore_index=True)

        # A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "must've": "must have",
            "mustn't": "must not",
            "needn't": "need not",
            "oughtn't": "ought not",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "that'd": "that would",
            "that's": "that is",
            "there'd": "there had",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where'd": "where did",
            "where's": "where is",
            "who'll": "who will",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are"
        }



        stemmer = nltk.stem.PorterStemmer()
        ps = PorterStemmer()

        # Remove unwanted characters,
        # and format the text to create fewer nulls word embeddings

        def clean_text(text, remove_stopwords=True):

            # Convert words to lower case
            text = text.lower()

            # Replace contractions with their longer forms
            if True:
                text = text.split()
                new_text = []
                for word in text:
                    if word in contractions:
                        new_text.append(contractions[word])
                    else:
                        new_text.append(word)
                text = " ".join(new_text)

            # Format words and remove unwanted characters
            text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\<a href', ' ', text)
            text = re.sub(r'&amp;', '', text)
            text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
            text = re.sub(r'<br />', ' ', text)
            text = re.sub(r'\'', ' ', text)
            text = re.sub("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"  # dingbats
                          u"\u3030"
                          "]+", '', text)

            # Tokenize each word
            text = nltk.WordPunctTokenizer().tokenize(text)

            return text


        timeline_df_RT['clean_tweet'] = timeline_df_RT['tweet'].apply(lambda x: clean_text(x))
        friends_df['clean_name'] = friends_df['name'].apply(lambda x: clean_text(x))

        # Removing stopwords belonging to english language
        from nltk.corpus import stopwords
        def remove_stopwords(text):
            words = [w for w in text if w not in stopwords.words('english')]
            return words

        # Appliying the remove_stopwords function to our dataset
        # and creating a new column called no_stop_words
        timeline_df_RT['no_stop_words'] = timeline_df_RT['clean_tweet'].apply(lambda x: remove_stopwords(x))
        timeline_df_RT.head()


        # removing and replacing suffixes to get to the root form of the word,
        # which is called the stem for instance cats - cat, wolves - wolv
        def lemmatizer_text_func(text):
            lemmatizer = nltk.stem.WordNetLemmatizer()
            timeline_df_RT['lemmatized_text'] = list(map(lambda word:
                                                         list(map(lemmatizer.lemmatize, word)),
                                                         timeline_df_RT.no_stop_words))

        lemmatizer_text_func(timeline_df_RT.no_stop_words)




        timeline_df_RT['liststring'] = [','.join(map(str, l)) for l in timeline_df_RT['lemmatized_text']]

        friends_df['clean_name'] = [''.join(map(str, l)) for l in friends_df['clean_name']]



        timeline_df_RT['new_full_string'] = ' '.join(timeline_df_RT["liststring"])
        friends_df['all_names'] = ' '.join(friends_df["clean_name"])
        timeline_df_RT['new_full_string'] = timeline_df_RT['new_full_string'].astype(str)
        timeline_df_RT['new_full_string'] = timeline_df_RT['new_full_string'].str.replace(' ',',')
        friends_df['all_names'] = friends_df['all_names'].str.replace(' ',',')



        new = new_movieDF[new_movieDF['Rating'] >= 7.0]
        new = new.fillna('')


        new['string_lemmatized_text'] = [','.join(map(str, l)) for l in new['lemmatized_text']]


        new['Genre'] = [','.join(map(str, l)) for l in new['Genre']]
        new['Genre']=new['Genre'].str.replace(',,', ',')


        def clean_data(x):
            return str.lower(x)

        new['Capital_Title'] = new['Title']
        a = new

        features=['Title','Director','Actor','Genre','string_lemmatized_text']
        new=new[features]


        for feature in features:
            new[feature] = new[feature].apply(clean_data)


        def create_soup(x):
            return x['Title'] + ' ' + x['Director'] + ' ' + x['Actor'] + ' ' + x['Genre'] + ' ' + x['string_lemmatized_text']


        new['soup'] = new.apply(create_soup, axis=1)

        new['Capital_Title'] = a['Capital_Title']

        twitter_movie_DF_predictor = new

        twitter_movie_DF_predictor = pd.DataFrame(twitter_movie_DF_predictor)

        timeline_df_RT['all_names'] = friends_df['all_names']

        timeline_df_RT['random'] = timeline_df_RT['new_full_string'] + " ," +timeline_df_RT['all_names']

        def new_clean_text(text, remove_stopwords=True):

            # Format words and remove unwanted characters
            text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\<a href', ' ', text)
            text = re.sub(r'&amp;', '', text)
            text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
            text = re.sub(r'…', ' ', text)
            text = re.sub(r'’', ' ', text)
            text = re.sub(r'—', ' ', text)
            text = re.sub(r'”', ' ', text)
            text = re.sub(r'<br />', ' ', text)
            text = re.sub(r'rt', ' ', text)
            text = re.sub(r'“', ' ', text)

            text = re.sub(r'\'', ' ', text)

            text = nltk.WordPunctTokenizer().tokenize(text)

            return text

        timeline_df_RT = timeline_df_RT.head(1)

        timeline_df_RT['random'] = timeline_df_RT['random'].apply(new_clean_text)

        timeline_df_RT['random'] = [','.join(map(str, l)) for l in timeline_df_RT['random']]

        twitter_movie_DF_predictor = twitter_movie_DF_predictor.append({'soup' : timeline_df_RT['random'][0],'Title' : timeline_df_RT['author'][0]} , ignore_index=True)

        vectorizer = TfidfVectorizer()
        count_matrix = vectorizer.fit_transform(twitter_movie_DF_predictor['soup'])
        cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

        twitter_movie_DF_predictor = twitter_movie_DF_predictor.reset_index()
        indices2 = pd.Series(twitter_movie_DF_predictor.index, index=twitter_movie_DF_predictor['Title'])

        def twitter_(title, cosine_sim2=cosine_sim2):
            idx = indices2[title]

            # Get the pairwsie similarity scores of all movies with that movie
            sim_scores = list(enumerate(cosine_sim2[idx]))

            # Sort the movies based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the scores of the 10 most similar movies
            sim_scores = sim_scores[1:40]

            # Get the movie indices
            movie_indices = [i[0] for i in sim_scores]

            # Return the top 10 most similar movies
            return twitter_movie_DF_predictor['Title'].iloc[movie_indices][0:number50]

        (st.write(twitter_(timeline_df_RT['author'][0], cosine_sim2)))





