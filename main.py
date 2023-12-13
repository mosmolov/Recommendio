import streamlit as st
import pandas as pd
st.set_page_config(
        page_title="Recommendio",
        page_icon="musical_note",
        layout="wide",
    )
from models import get_KNN_recommendations, get_CosineSim_Recommendations
from fuzzywuzzy import process
import webbrowser

data = pd.read_csv('algorithms/genres_v2.csv', low_memory=False)
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
def open_link(url):
    webbrowser.open_new_tab(url)
# add sidebar with clickable table of contents to navigate to different headers
# make sidebar contents clickable
st.sidebar.markdown(f"<h1 style='margin-top: 0px;'>Spotify Recommendation Algorithm</h1>", unsafe_allow_html=True)
st.sidebar.markdown(f"<a href='#introduction-background' style='text-decoration: none; color: #90EE90;'><h2>Introduction/Background</h2></a>", unsafe_allow_html=True)
#st.sidebar.markdown(f"<hr style='border: 1px solid white; margin: 1px;'>", unsafe_allow_html=True)
st.sidebar.markdown(f"<a href='#problem-definition' style='text-decoration: none; color: #90EE90;'><h2>Problem Definition</h2></a>", unsafe_allow_html=True)
# st.sidebar.markdown(f"<hr style='border: 1px solid white; margin: 1px;'>", unsafe_allow_html=True)
st.sidebar.markdown(f"<a href='#dataset' style='text-decoration: none; color: #90EE90;'><h2>Dataset</h2></a>", unsafe_allow_html=True)
st.sidebar.markdown(f"<a href='#methods' style='text-decoration: none; color: #90EE90;'><h2>Methods</h2></a>", unsafe_allow_html=True)
st.sidebar.markdown(f"<a href='#metrics' style='text-decoration: none; color: #90EE90;'><h2>Metrics</h2></a>", unsafe_allow_html=True)
st.sidebar.markdown(f"<a href='#results' style='text-decoration: none; color: #90EE90;'><h2>Results</h2></a>", unsafe_allow_html=True)
st.sidebar.markdown(f"<a href='#gantt-chart' style='text-decoration: none; color: #90EE90;'><h2>Gantt Chart</h2></a>", unsafe_allow_html=True)
st.sidebar.markdown(f"<a href='#contribution-table' style='text-decoration: none; color: #90EE90;'><h2>Contribution Table</h2></a>", unsafe_allow_html=True)
st.sidebar.markdown(f"<a href='#references' style='text-decoration: none; color: #90EE90;'><h2>References</h2></a>", unsafe_allow_html=True)

'# Spotify Recommendation Algorithm' 

st.header("Get Recommendations")
# allow user to search for a song
user_input = st.text_input("Enter a song name", "XO Tour Llif3")
typeOfModel = st.selectbox("Select a model", ("KNN", "Cosine Similarity"))
if st.button("Get Recommendations"):
    st.session_state.show_results = True
    try:
        recommendations = get_KNN_recommendations(user_input) if typeOfModel == "KNN" else get_CosineSim_Recommendations(user_input)
        song_details = []
        for name in recommendations:
            # Find the song in the dataset
            song = data[data['song_name'] == name]
            # select only first row  
            song = song.iloc[0]
            songUrl = 'https://open.spotify.com/track/' + song["id"]
            # Extract artist, genre, and other details
            genre = song['genre']
            if name.count('$') == 2:
                name = name.replace('$', '\\$')
            if genre and songUrl:
                song_details.append({'song_name': name, 'genre': genre, 'url': songUrl})
            elif genre:
                song_details.append({'song_name': name, 'genre': genre})
            elif songUrl:
                song_details.append({'song_name': name, 'url': songUrl})
            print(song_details)
        col1, col2, col3, col4, col5 = st.columns(5)
        if st.session_state.show_results:
            for i in range(1, len(song_details) + 1):
                spotifyButton_html = f"""
                            <style>
                                .custom-button {{
                                    color: white;
                                    border: 2px solid white;
                                    padding: 10px 20px;
                                    border-radius: 15px;
                                    text-align: center;
                                    text-decoration: none;
                                    display: inline-block;
                                    font-size: 16px;
                                    margin: 4px 2px;
                                    cursor: pointer;
                                    background-color: black; /* Black background */
                                    transition-duration: 0.4s;
                                }}
                                .custom-button:hover {{
                                    color: #1DB954; /* Text turns green on hover */
                                    border: 2px solid #1DB954; /* Border turns green on hover */
                                    background-color: black; /* Background stays black */
                                }}
                            </style>
                        <a href="{song_details[i-1]['url']}" target="_blank">
                            <button class="custom-button">Listen on Spotify</button>
                        </a>
                            """
                if i % 5 == 1:
                    with col1:                        
                        with st.container():
                            st.subheader(f'{song_details[i-1]["song_name"]}')
                            st.text(f"Genre: {song_details[i-1]['genre']}")
                            st.markdown(spotifyButton_html, unsafe_allow_html=True)
                            # # add separator at standard height
                            # st.markdown("---")
                elif i % 5 == 2:
                    with col2:
                        with st.container():
                            st.subheader(f'{song_details[i-1]["song_name"]}')
                            st.text(f"Genre: {song_details[i-1]['genre']}")
                            st.markdown(spotifyButton_html, unsafe_allow_html=True)
                            # st.markdown("---")
                elif i % 5 == 3:
                    with col3:
                        with st.container():
                            st.subheader(f'{song_details[i-1]["song_name"]}')
                            st.text(f"Genre: {song_details[i-1]['genre']}")
                            st.markdown(spotifyButton_html, unsafe_allow_html=True)
                            # st.markdown("---")
                elif i % 5 == 4:
                    with col4:
                        with st.container():
                            st.subheader(f'{song_details[i-1]["song_name"]}')
                            st.text(f"Genre: {song_details[i-1]['genre']}")
                            st.markdown(spotifyButton_html, unsafe_allow_html=True)
                            # st.markdown("---")
                else:
                    with col5:
                        with st.container():
                            st.subheader(f'{song_details[i-1]["song_name"]}')
                            st.text(f"Genre: {song_details[i-1]['genre']}")
                            st.markdown(spotifyButton_html, unsafe_allow_html=True)
                                
                            # st.markdown("---")
                
    except:
        st.error("Song not found. Did you mean to input one of these songs?")
        if user_input:
            songNames = data['song_name'].dropna().tolist()
            suggestions= process.extract(user_input, songNames, limit=5)
            # remove duplicates from suggestions
            suggestions = list(dict.fromkeys(suggestions))
            for suggestion in suggestions:
                st.write(suggestion[0])
    
    
st.header("Introduction/Background") #A quick introduction of your topic and mostly literature review of what has been done in this area. 
'Finding new music is something that everyone struggles with. We aim to quell the struggle using machine learning to suggest songs similar to user taste, by analyzing existent songs on Spotify according to their features of danceability, energy, and others.'
'By doing this, we will be able to recommend songs to users according to several filters that they select, which will be according to mood, vocals, genre, and others.'
'Currently, there has been lots of research into the Spotify Recommendation Algorithm and improving it and currently a lot of research is being done in regards to collaborative filtering and natural language processing to further optimize the song recommendations [1, 2, 3].'
'To accomplish this, we plan on using a dataset containing information about thousands of songs on spotify of various genres, in order to find similar traits amongst certain songs'
st.header("Problem Definition")
'Music is apart of almost everyones life. But often times, we get bored of listening to the same songs over and over again. One of the most difficult tasks if finding new songs to listen to. So, we want to solve this problem by creating a song recommendation model for music enthusiasts to find brand new music.'
'Most people do not know how to find new music, they only know what mood or genre they want to listen to. This project fixes that need by simply asking the user for certain baseline preferences, and recommending the right music accordingly.'
st.header("Dataset")
'We will be using a [Kaggle dataset](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data) of Spotify songs and playlists which has data on each song\'s metrics according to its danceability, energy, speechiness, instrumentalness, and more [4].'
# Read in data
df = pd.read_csv('algorithms/genres_v2.csv', low_memory=False)
st.dataframe(df[20:30])

'**Danceability** - Combines tempo, rhythm stability, beat strength, and overall regularity to give a value for how \"danceable\" a song is.'
'**Energy** - A meeasure of intensity and activity. The more energy the more fast, loud, and noisy.'
'**Key** - The key of the track.'
'**Loudness** - How loud the song is in decibels (dB) averaged over the duration of the song.'
'**Speechiness** - Detects the presence of spoken words in a track.'
'**Acousticness** - A confidence measure of the tracks acoustics.'
'**Instrumentalness** - A predicition of whether a track contains vocals.'
'**Liveness** - This detects if an audience was present in the recording of the track.'
'**Valence** - A measure, made by seeing the valence, describing the musical positiveness conveyed by a track.'
'**Tempo** - The overall tempo of a track in beats per minute (BPM) averaged over time.'
'Since there are a lot of features, it\'s hard to visualize the data. We will use PCA to reduce the dimensionality of the data and plot it in 2D.'


'To identify relevant features, we use a correlation matrix with a heat map to visualize the correlation between the features. We can see that there is a strong correlation between danceability and energy, and a strong negative correlation between acousticness and energy. We will use these features to cluster the songs.'
st.image('images/heatmap.png', width=500)

'From the heatmap, we can see that there is a strong positive correlation between energy and loudness, as well as a strong negative correlation between energy and acousticness. There is also some correlation between danceability and valence.'
'We can further visualize this by creating linear regression plots for each pair of features.'
col1, col2, col3 = st.columns(3)
with col1:
    st.image('images/energy_acousticness.png')
with col2:
    st.image('images/energy_loudness.png')
with col3: 
    st.image('images/danceability_valence.png')

'For all of these models we needed to do data preprocessing. We had to remove some features from the models as they were not in scope with what we were trying to do. For KNN specifically, we also took into account the genre, something we did not do for the other models. This may have contributed to KNN\'s success.'

st.header("Methods") # What algorithms or methods are you going to use to solve the problems. (Note: Use existing packages/libraries)'
'First, we use scikit to implement a cosine similarity model to quantify the similarity between different songs. In theory, this means representing each song as a vector in 5D space (each dimension is a feature of the song).'

''

'The next step is to normalize each vector to a length of 1, so that the similarity measurement is unrelated to the actual vector\'s length.'
'Lastly, we use the cosine similarity formula to find the angle (similarity) between two different vectors.'
#INSERT IMAGE 
'We are going to use scikit learn to implement a bottom-up agglomerative hierarchical clustering algorithm to cluster the songs into groups based on their features. We will then use the clusters to recommend songs to users based on their preferences.'
'First, we load the dataset using Pandas. We perform initial data processing, which involves removing columns that are not necessary for the clustering process. This step is crucial for focusing on relevant features that contribute to understanding song similarities.'

'Next, we standardize the features using a StandardScaler. This essentially rescales the features so that they have a mean of 0 and a standard deviation of 1, which is important because it normalizes the range of independent variables or features of the data. This makes sure that each feature contributes equally to the analysis and prevents features with larger scales from dominating the distance calculations used in clustering.'

'After standardization, we used PCA for dimensionality reduction. Since we have 10 features, we can use PCA to reduce the dimensionality of our dataset to 3 dimensions and narrow down the optimal directions to project and evaluate the features on. The goal of this is to retain 95% of the variance, keeping the most important features. This step helps us for clustering, because we reduce noise and computation time.'

'Next, we conduct clustering analysis to gain an understanding of the optimal number of clusters (k) for the dataset. To identify the ideal value of k, we explore a range of potential cluster sizes, spanning from 5000 to 6000 clusters. For each candidate cluster size, we apply Agglomerative Clustering to the PCA-transformed data. The Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index are then calculated and recorded. These metrics serve as quantitative measures of clustering quality:'

'To visualize the performance of the clustering algorithm, we create three plots to display the three different scores with varying clusters k.'

'Now, using these metrics, we specify our Agglomerative Clustering to have the optimal amount of clusters. To facilitate song recommendations, we construct a Nearest Neighbors model, crucial for identifying the nearest cluster to an input song, a key step in the recommendation process.'

'Now, using the optimal number of clusters determined from the silhouette scores, we applied Agglomerative Clustering, a hierarchical clustering that builds nested clusters by splitting them successively. This clustering model will group the songs in our dataset into clusters based on their similarities in the feature space.'
col4, col5, col6 = st.columns(3)
with col4:
    st.image('images/CHForClusters.png')
with col5:
    st.image('images/dbForClusters.png')
with col6:
    st.image('images/silhouettescores.png')

'Another model we used was K-Nearest Neighbors (KNN). KNN is an algorithm for classification and regression tasks. In the context of our recommender, we used KNN as follows. First, we must chose a K, which represents the number of nearest neighbors to consider when making a prediction. From there find its neighbors, in our specific example we have 6 neighbors. We then measured the distance from K to each of its neighbors. We did this with different typed of distance algorithms such as Euclidean distance, Manhattan distance, Cosine distance, and Minkowski distance. We then compared these distances to see overlap in recommendations and see which distance algorithms produced similar results and which ones are uniquely identifying recommendations. Below is how we saw the overlap.'
st.image('images/jawn.png', width=500)

'The final model we used was DBScan. However, as we tried to implement it in the context of our recommender, there was too much noise so it was not a viable model. We tried to tweak the hyperparameters to minimize noise but the noise still existed as we kept testing and thus we could not use it. '


st.header("Metrics") # Metrics for how to see accuracy of recommendation system.
# 'To see how accurate our model is and whether people are getting the correct recommendations, we must have a set of metrics.'
# 'We will seperate the metrics into multiple types of evaluations.'
# 'The first type is offline evaluation. The offline testing procedure has two strategies - Train-Test Split and Cross-Validation.'
# 'In our first model we will be using Train-Test split as it is more simple. The dataset is split randomly in to 70% - 80% for training and the rest for testing. Then we purely train the model on the training set, picking random songs and recommending k songs similar to that song. This is compared to our manual selection of k songs similar to the original song. Then the trained model is used to predict recommendations for the songs in the test set, and its predictions are compared against the actual songs to evaluate performance metrics such as precision. Precision measures the proportion of recommmended songs in the top-k set that are relevant. We determine how many of the k songs in our opinion is similar to the ones recommended by our model. We then calculate the fraction of these matches over k. A high precision value means that when the system presents k items a large portion of these items are typically relevant to the user, indicating the system is successful in filtering out irrelevant items from the recommendation list. We are not considering recall, as there are many more than 5 songs that could be recommended for every song and at some point it is based on subjective taste.'
# 'For our second model, we will be using a more robust form of train-test split known as Cross-Validation. To do so, we will need to partition the dataset into k equal folds. We will select one fold as the test, and the rest is the training. Validate the model in the test folds using precision at k for each iteration. After Cross-Validation iterates across k folds, aggregate the performance to get a compelete view of the models performance. '
# 'Another type of evaluation is the conversion rate. The conversion rate will track the proporition of receommendations that lead to the user either adding the song to their liked songs/playlist or simply streaming the song for a measured time.'
# 'We will use the conversion rate for A/B Testing. A/B testing involves sowing recommendations from the new model to one group of users (test group) and recommendations from the current model to another (control) group. The conversion rate will analyze real world user behavior, and compare it to the control group to help improve the model.'

'To see how accurate our model is and whether people are getting the correct recommendations, we must have a set of metrics.'
'We will view the accuracy of the two models we created by these metrics. First we will look at cosine similarity.'
'Evaluating the accuracy of the cosine similarity model is difficult because we do not have user input to compare the recommendations to. However, we can calculate intra-list similarity between the recommended songs to see how similar they are to each other. We can also calculate the average similarity between the recommended songs and the input song to see how similar they are to the input song.'
'To do this, we calculate the pair-wise similarity between each song in the recommendation list. Then we take the average of these to see the overall intra-list similarity score. This shows us whether the recommendations are similar to each other, and whether the model is consistent in its recommendations.'
code = '''def calculate_intra_list_similarity(recommendations):
    recommendations_df = pd.DataFrame({
        'Song Name': [rec[0] for rec in recommendations],
        'Similarity Score': [rec[1] for rec in recommendations]
    })
    totalSimilarity = 0
    totalComparisons = 0
    for i in range(0, len(recommendations) - 1):
        for j in range(i + 1, len(recommendations)):
            song1 = recommendations[i][0]
            song2 = recommendations[j][0]
            song1_index = df_unique_songs.index[df_unique_songs['song_name'] == song1].tolist()
            song2_index = df_unique_songs.index[df_unique_songs['song_name'] == song2].tolist()
            if song1_index and song2_index and song1_index[0] < len(cosine_sim) and song2_index[0] < len(cosine_sim[song1_index[0]]):
                if (cosine_sim[song1_index[0]][song2_index[0]]):
                    totalSimilarity += cosine_sim[song1_index[0]][song2_index[0]]
                    totalComparisons += 1
    return totalSimilarity / totalComparisons if totalComparisons > 0 else 0'''
       
st.code(code, language='python')
'From calculating the intra-list similarity score for the recommendations for \'XO Tour Llif3\' by Lil Uzi Vert, we get an intra-list similarity score of 0.8214426102366348, which shows that overall the recommendations are similar to each other.'

'Now to check consistency, we can get recommendations for the first 10 songs in the dataset, and then average their intra-list similarity scores to get a more conclusive result.'
'After doing this, we get an average intra-list similarity score of 0.8664253309369923, which shows good consistency amongst recommendations.'

'Next we will look at the accuracy of the agglomerative clustering.'
'The Silhouette Score is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). For the cluster size of 5000, the Silhouette Score is approximately 0.3004, indicating a moderate level of cluster quality. It suggests that the songs within each cluster have some degree of similarity, but there is room for improvement. As we increase the number of clusters to 5250, the Silhouette Score remains similar at around 0.3005, signifying a consistent level of clustering quality. This suggests that the choice of 5250 clusters does not significantly impact the cohesion and separation of songs within clusters.'

'The Davies-Bouldin Index is another clustering evaluation metric that measures the average similarity between each cluster and its most similar cluster. A lower index indicates better clustering, with lower values signifying greater separation between clusters. At a cluster size of 5000, the Davies-Bouldin Index is approximately 0.673, suggesting that the clusters are reasonably well-separated, but there is still room for improvement. As the number of clusters increases to 5250, the Davies-Bouldin Index decreases slightly to around 0.657, indicating that the clusters become slightly better separated. This aligns with the Silhouette Score\'s findings.'

'The Calinski-Harabasz Index, which is also referred to as the Variance Ratio Criterion, measures the ratio of between-cluster variance to within-cluster variance. Higher values indicate better clustering, with larger values indicating more compact and well-separated clusters. For a cluster size of 5000, the Calinski-Harabasz Index is approximately 1312.76. This suggests that there is a moderate level of separation between clusters, but it may not be optimal. As we increase the number of clusters to 5250, the Calinski-Harabasz Index also increases to around 1331.70, indicating that the clustering quality improves slightly with more clusters.'

'The clustering results based on the Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index suggest that the choice of 5250 clusters offers a good balance between cohesion and separation within clusters. While the Silhouette Score remains fairly consistent across different cluster sizes, the Davies-Bouldin Index and Calinski-Harabasz Index show slight improvements with a cluster size of 5250. These metrics collectively indicate that the clustering quality is moderate, with room for further optimization. It\'s important to note that the choice of the number of clusters should also consider the practicality of song recommendations and user preferences. To enhance clustering quality, we can explore alternative clustering algorithms, feature engineering, or different dimensionality reduction techniques.'

'Finally, we wanted to keep a consistent metric between the two models so we used intra similarity list for this model as well. Similar to what we did for the cosine simialrity, we calculated the pair-wise similarity between each song in the recommendation list and took the average of these to see the overall intra-list similarity score.'
'The code is as follows:'
code = '''def calculate_intra_list_similarity(recommendations, spotify_df, standard_scaler, pca):
    total_distance = 0
    total_comparisons = 0

    for i in range(len(recommendations) - 1):
        for j in range(i + 1, len(recommendations)):
            song1_name = recommendations[i]
            song2_name = recommendations[j]

            song1_features = spotify_df[spotify_df['song_name'] == song1_name].drop(columns=['song_name', 'cluster']).iloc[0]
            song2_features = spotify_df[spotify_df['song_name'] == song2_name].drop(columns=['song_name', 'cluster']).iloc[0]

            preprocessed_song1 = preprocess_song(song1_features, standard_scaler, pca)
            preprocessed_song2 = preprocess_song(song2_features, standard_scaler, pca)

            distance = np.linalg.norm(preprocessed_song1 - preprocessed_song2)
            total_distance += distance
            total_comparisons += 1

    return total_distance / total_comparisons if total_comparisons > 0 else 0'''
st.code(code, language='python')
'From calculating the intra-list similarity score for the recommendations for \'XO Tour Llif3\' by Lil Uzi Vert, we get an intra-list similarity score of 0.07287909464867108, which shows that overall the recommendations are not similar to each other.'
# st.image('images/intraViz.tiff', width=500)

'As we continued to implement new models, we still used intra-list similarity to measure the score to see how accurate it was. We did this as its a metric we could use for all of the models to compare them to one another. Here are the three models intra-list similarities:'
'Agglomerative Clustering'
st.image('images/aglom.png', width=500)
'Cosine Similarity'
st.image('images/cosine.png', width=500)
'KNN'
st.image('images/knn.png', width=500)

'We then compared these three intra-list similarities with this bar graph:'
st.image('images/comparison.png', width=500)

'As this comparison shows, our best model was KNN, then Cosine Similarity, and finally Agglomerative Clustering.'

st.header("Results")
'Given these metrics, we were able to analyze that Cosine Similarity is signifcantly more accurate in recommending a similar song to one given than the Agglomerative Clustering model. We can conclude this as we have a metric that was used for both models. The intra-list similarity metric is a measure used to evaluate the homogeneity or consistency within a set of items, such as recommendations in a recommendation system. In the context of song recommendations, for instance, it quantifies how similar the recommended songs are to each other. This metric is particularly useful in scenarios where it\'s desirable for the items in a list (like songs, products, articles, etc.) to be similar or related in some way.'

'We found the intra-list similarity for the Cosine Similarity model for the song \'XO Tour Llif3\' by Lil Uzi Vert was 0.8664253309369923, while for the same song the Agglomerative Clustering model had one of 0.07287909464867108.'

'After comparing this with the new KNN model, we found an intra-similarity of 0.9999999988248943, thus making it the most accurate model for giving song recommendations.'

st.header("Gantt Chart")
st.markdown('[Open Chart](https://docs.google.com/spreadsheets/d/128ocUWtq5-0vj90tC-5R_a0zHKy6ZOHwDNsW2s9xa2U/edit?usp=sharing)')

st.header("Contribution Table")
# Create a list of dictionaries with each member's name and their contribution
contributions = [
    {'Name': 'Adhish Rajan', 'Contribution': 'Identifying methods/algorithms, Agglomerative Clustering Model, Implemented additional models'},
    {'Name': 'Michael Osmolovskiy', 'Contribution': 'Creating Github repository, finding dataset,  writing introduction, problem definition, methods, identifying metrics, Data Visualization, Implemented additional models'},
    {'Name': 'Abhinav Vishnuvajhala', 'Contribution': 'Creating presentation slides for video and recording video proposal, Website Beefing, Metrics, Presentation, Video'},
    {'Name': 'Arin Khanna', 'Contribution': 'Creating Gantt Chart, Website, Beefing, Creating Metrics for models, Analyzing Results, Implement visualizations in website'},
    {'Name': 'Vedesh Yadlapalli', 'Contribution': 'Researching and identifying the problem being solved, writing proposal, finding peer-reviewed articles, Cosine Similarity Model, Metrics visualizations'},
]
df = pd.DataFrame(contributions)
# Display the table using st.table
st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

st.header("References")
st.markdown('[1] Jie and Han. [A music recommendation algorithm based on clustering and latent factor model.](https://www.matec-conferences.org/articles/matecconf/pdf/2020/05/matecconf_cscns2020_03009.pdf)')
st.markdown('[2] Pérez-Marcos and López. [Recommender System Based on Collaborative Filtering for Spotify"s Users](https://www.researchgate.net/publication/318511102_Recommender_System_Based_on_Collaborative_Filtering_for_Spotify"s_Users)')
st.markdown('[3] Schedl, Zamani, Chen, Deldjoo, and Elahi. [Current Challenges and Visions in Music Recommender Systems Research](https://browse.arxiv.org/pdf/1710.03208.pdf)')
st.markdown('[4] Samoshyn. [Dataset of songs in Spotify](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data)')










