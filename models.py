import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
    