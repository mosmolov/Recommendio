import streamlit as st
import pandas as pd
st.set_page_config(
        page_title="Recommendio",
        page_icon="musical_note",
        layout="wide",
    )
'# Spotify Recommendation Algorithm' 

'## Introduction/Background:' #A quick introduction of your topic and mostly literature review of what has been done in this area. 
'Finding new music is something that everyone struggles with. We aim to quell the struggle using machine learning to suggest songs similar to user taste, by analyzing existent songs on Spotify according to their features of danceability, energy, and others.'
'By doing this, we will be able to recommend songs to users according to several filters that they select, which will be according to mood, vocals, genre, and others.'
'Currently, there has been lots of research into the Spotify Recommendation Algorithm and improving it and currently a lot of research is being done in regards to collaborative filtering and natural language processing to further optimize the song recommendations [1, 2, 3].'
'To accomplish this, we plan on using a dataset containing information about thousands of songs on spotify of various genres, in order to find similar traits amongst certain songs'
'## Problem Definition: '
'Many people enjoy listening to music, however they get bored of listening to the same songs over and over again. We aim to solve this problem by recommending songs to users based on their preferences.'
'Most people do not know how to find new music, they only know what mood or genre they want to listen to. This project fixes that need by simply asking the user for certain baseline preferences, and recommending the right music accordingly.'
'## Data Collection: '
'We will be using a [Kaggle dataset](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data) of Spotify songs and playlists which has data on each song\'s metrics according to its danceability, energy, speechiness, instrumentalness, and more [4].'
# Read in data
df = pd.read_csv('genres_v2.csv')
st.dataframe(df[20:30])
'Since there are a lot of features, it\'s hard to visualize the data. We will use _____ to reduce the dimensionality of the data and plot it in 2D.'


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
'## Methods:' # What algorithms or methods are you going to use to solve the problems. (Note: Use existing packages/libraries)'
'We are going to use scikit learn to implement a bottom-up agglomerative hierarchical clustering algorithm to cluster the songs into groups based on their features. We will then use the clusters to recommend songs to users based on their preferences.'
'First, we will cluster them by individual features likes danceability and energy, gradually building up the hierarchy until at the top level, we cluster them by genre.'
'## Potential Results and Discussion:' #Discuss about what type of quantitative metrics your team plan to use for the project (i.e. ML Metrics).
'Since we know the true cluster assignments of the songs based on their data and genres, we will use the scikit.metrics module to implement Rand Index as our evaluation metric.'
'This ignores permutations and requires knowledge of ground truth classes, which we have in this case, given the song data and genres.'
'## Metrics:' # Metrics for how to see accuracy of recommendation system.
'To see how accurate our model is and whether people are getting the correct recommendations, we must have a set of metrics.'
'We will seperate the metrics into multiple types of evaluations.'
'The first type is offline evaluation. The offline testing procedure has two strategies - Train-Test Split and Cross-Validation.'
'In our first model we will be using Train-Test split as it is more simple. The dataset is split randomly in to 70% - 80% for training and the rest for testing. Then we purely train the model on the training set, picking random songs and recommending k songs similar to that song. This is compared to our manual selection of k songs similar to the original song. Then the trained model is used to predict recommendations for the songs in the test set, and its predictions are compared against the actual songs to evaluate performance metrics such as precision. Precision measures the proportion of recommmended songs in the top-k set that are relevant. We determine how many of the k songs in our opinion is similar to the ones recommended by our model. We then calculate the fraction of these matches over k. A high precision value means that when the system presents k items a large portion of these items are typically relevant to the user, indicating the system is successful in filtering out irrelevant items from the recommendation list. We are not considering recall, as there are many more than 5 songs that could be recommended for every song and at some point it is based on subjective taste.'
'For our second model, we will be using a more robust form of train-test split known as Cross-Validation. To do so, we will need to partition the dataset into k equal folds. We will select one fold as the test, and the rest is the training. Validate the model in the test folds using precision at k for each iteration. After Cross-Validation iterates across k folds, aggregate the performance to get a compelete view of the models performance. '
'Another type of evaluation is the conversion rate. The conversion rate will track the proporition of receommendations that lead to the user either adding the song to their liked songs/playlist or simply streaming the song for a measured time.'
'We will use the conversion rate for A/B Testing. A/B testing involves sowing recommendations from the new model to one group of users (test group) and recommendations from the current model to another (control) group. The conversion rate will analyze real world user behavior, and compare it to the control group to help improve the model.'
'## Results:'
''


'## Gantt Chart'

st.markdown('[Open Chart](https://docs.google.com/spreadsheets/d/128ocUWtq5-0vj90tC-5R_a0zHKy6ZOHwDNsW2s9xa2U/edit?usp=sharing)')

'## Contribution Table'

# Create a list of dictionaries with each member's name and their contribution
contributions = [
    {'Name': 'Adhish Rajan', 'Contribution': 'Identifying methods/algorithms, Agglomerative Clustering Model'},
    {'Name': 'Michael Osmolovskiy', 'Contribution': 'Creating Github repository, finding dataset,  writing introduction, problem definition, methods, identifying metrics, Data Visualization'},
    {'Name': 'Abhinav Vishnuvajhala', 'Contribution': 'Creating presentation slides for video and recording video proposal, Website Beefing, Metrics'},
    {'Name': 'Arin Khanna', 'Contribution': 'Creating Gantt Chart, Website, Beefing, Creating Metrics for models'},
    {'Name': 'Vedesh Yadlapalli', 'Contribution': 'Researching and identifying the problem being solved, writing proposal, finding peer-reviewed articles, Cosine Similarity Model'},
]
df = pd.DataFrame(contributions)
# Display the table using st.table
st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

'## References'
st.markdown('[1] Jie and Han. [A music recommendation algorithm based on clustering and latent factor model.](https://www.matec-conferences.org/articles/matecconf/pdf/2020/05/matecconf_cscns2020_03009.pdf)')
st.markdown('[2] Pérez-Marcos and López. [Recommender System Based on Collaborative Filtering for Spotify"s Users](https://www.researchgate.net/publication/318511102_Recommender_System_Based_on_Collaborative_Filtering_for_Spotify"s_Users)')
st.markdown('[3] Schedl, Zamani, Chen, Deldjoo, and Elahi. [Current Challenges and Visions in Music Recommender Systems Research](https://browse.arxiv.org/pdf/1710.03208.pdf)')
st.markdown('[4] Samoshyn. [Dataset of songs in Spotify](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data)')


