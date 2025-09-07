import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import pickle

df = pd.read_csv(r"C:\Users\Deeksha\Desktop\TrendPulse\Tp\ex.csv")

df.dropna(inplace=True)

df.isnull().sum()

df = df.drop_duplicates()

df.duplicated().sum()

df.shape

df.head()

df['User-Rating']

l = []
for i in df['User-Rating']:
    l.append(i[:3])

df['User-Rating'] = l

df['Album/Movie'] = df['Album/Movie'].str.replace(' ', '')
df['Singer/Artists'] = df['Singer/Artists'].str.replace(' ', '')
df['Singer/Artists'] = df['Singer/Artists'].str.replace(',', ' ')

df['tags'] = df['Singer/Artists'] + ' ' + df['Genre'] + ' ' + df['Album/Movie'] + ' ' + df['User-Rating']
df['tags'][0]

new_df = df[['Song-Name', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

cv = CountVectorizer(max_features=2000)

vectors = cv.fit_transform(new_df['tags']).toarray()

vectors.shape

cv.get_feature_names_out()

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])

new_df.rename(columns={'Song-Name': 'title'}, inplace=True)

def lst():
    print(df['Genre'].unique())
    return 0

def recommend(genre):
    # Concatenate tags to create a combined tag
    df['tags'] = df['Singer/Artists'] + ' ' + df['Genre'] + ' ' + df['Album/Movie'] + ' ' + df['User-Rating']
    new_df = df[['Song-Name', 'tags']]

    # Filter dataframe based on genre
    genre_df = new_df[new_df['tags'].str.contains(genre, case=False)]

    # Calculate similarity based on tags
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # similarity_matrix = cosine_similarity(TfidfVectorizer.fit_transform(genre_df['tags']))

    similarity = cosine_similarity(vectors)

    # Get the index of the input genre
    genre_index = genre_df.index[0]

    # Get distances/similarities of songs to the input genre
    distances = similarity[genre_index]

    # Shuffle the indices to get random selection
    random.shuffle(distances)

    # Get top 5 similar songs
    music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:5]

    # Print recommended songs
    s = []
    for i in music_list:
        s.append(genre_df.iloc[i[0]]['Song-Name'])
        print(genre_df.iloc[i[0]]['Song-Name'])
    return s


def recd_song(music):
    music_index = new_df[new_df['title'] == music].index[0]
    distances = similarity[music_index]
    music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:5]
    rs = []
    for i in music_list:
        rs.append(new_df.iloc[i[0]].title)
        print(new_df.iloc[i[0]].title)
    return rs


pickle.dump(new_df, open('musicrec.pkl', 'wb'))
pickle.dump(similarity, open('similarities.pkl', 'wb'))

if __name__ == '__main__':
    print("WELCOME TO TrendPulse - MusicPulse !!!")
    genRe = ' '
    song = ' '
    Songs = []
    print("TRENDING GENRE : ")
    genRe = input("FOR RECOMMENDATIONS BASED ON GENRE, SELECT YOUR FAVOURITE GENRE - ")
    Songs.append(recommend(genRe))
    song = input("FOR RECOMMENDATIONS BASED ON SONGS, SELECT YOUR FAVOURITE SONGS - ")
    Songs.append(recd_song(song))


