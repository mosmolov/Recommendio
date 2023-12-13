import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('algorithms/genres_v2.csv', low_memory=False)

def get_KNN_recommendations(input_song, metric="cosine", data=data):
    # Preprocessing the data
    # Assuming 'genre' as the target variable for classification
    X = data.select_dtypes(include=['float64', 'int64']).dropna()  # Selecting numerical features
    y = data.loc[X.index, 'genre']  # Corresponding genres
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Standardizing the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    input_index = data[data['song_name'] == input_song].index[0]

    input_features = X.iloc[input_index]
    input_features = input_features.values.reshape(1, -1)
    nn = NearestNeighbors(n_neighbors=6, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(input_features)
    nearest_neighbors = indices[0]

    recommended_songs = data.iloc[nearest_neighbors]['song_name']
    
    recommendations = recommended_songs[recommended_songs != input_song].tolist()
    return recommendations

def get_CosineSim_Recommendations(song_name, num_of_songs=5):
    df_unique_songs = data.drop_duplicates(subset='song_name', keep='first')
    # Normalize the feature values using Min-Max Scaler
    scaler = MinMaxScaler()
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'liveness', 'loudness', 'speechiness', 'valence', 'tempo', 'key']
    df_scaled = pd.DataFrame(scaler.fit_transform(df_unique_songs[features]), 
                            index=df_unique_songs.index, 
                            columns=features)
    # Compute the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(df_scaled)
    if song_name not in df_unique_songs['song_name'].values:
        print(f"The song '{song_name}' does not exist in the dataset. Please check for typos or use a different song.")
        return []

    # Find the index of the song that matches the song_name
    song_index = df_unique_songs.index[df_unique_songs['song_name'] == song_name].tolist()[0]
    sim_scores = list(enumerate(cosine_sim_matrix[song_index]))
    
    # Sort the songs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the first element if it is the song itself (similarity = 1)
    if sim_scores[0][1] == 1.0:
        sim_scores = sim_scores[1:num_of_songs+1]
    else:
        sim_scores = sim_scores[:num_of_songs]

    # Get the song indices and similarity scores
    song_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    # Fetch the song names using the indices
    recommended_songs = df_unique_songs.iloc[song_indices]
    
    return recommended_songs['song_name'].tolist()
    