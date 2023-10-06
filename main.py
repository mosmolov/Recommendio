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
'We will be using a [Kaggle dataset](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data) of Spotify songs and playlists which has data on each song''s metrics according to its danceability, energy, speechiness, instrumentalness, and more [4].'
'## Problem Definition: '
'Many people enjoy listening to music, however they get bored of listening to the same songs over and over again. We aim to solve this problem by recommending songs to users based on their preferences.'
'Most people do not know how to find new music, they only know what mood or genre they want to listen to. This project fixes that need by simply asking the user for certain baseline preferences, and recommending the right music accordingly.'
'## Methods:' # What algorithms or methods are you going to use to solve the problems. (Note: Use existing packages/libraries)'
'We are going to use scikit learn to implement a bottom-up agglomerative hierarchical clustering algorithm to cluster the songs into groups based on their features. We will then use the clusters to recommend songs to users based on their preferences.'
'First, we will cluster them by individual features likes danceability and energy, gradually building up the hierarchy until at the top level, we cluster them by genre.'
'## Potential Results and Discussion:' #Discuss about what type of quantitative metrics your team plan to use for the project (i.e. ML Metrics).
'Since we know the true cluster assignments of the songs based on their data and genres, we will use the scikit.metrics module to implement Rand Index as our evaluation metric.'
'This ignores permutations and requires knowledge of ground truth classes, which we have in this case, given the song data and genres.'

'## Gantt Chart'

st.markdown('[Open Chart](https://docs.google.com/spreadsheets/d/128ocUWtq5-0vj90tC-5R_a0zHKy6ZOHwDNsW2s9xa2U/edit?usp=sharing)')

'## Contribution Table'

# Create a list of dictionaries with each member's name and their contribution
contributions = [
    {'Name': 'Adhish Rajan', 'Contribution': 'Creating Gantt Chart and identifying methods/algorithms'},
    {'Name': 'Michael Osmolovskiy', 'Contribution': 'Creating Github repository, finding dataset, and writing introduction, problem definition, methods, and identifying metrics'},
    {'Name': 'Abhinav', 'Contribution': 'Creating presentation slides for video and recording video proposal'},
    {'Name': 'Arin Khanna', 'Contribution': 'Creating Gantt Chart'},
    {'Name': 'Vedesh Yadlapalli', 'Contribution': 'Researching and identifying the problem being solved, writing proposal, finding peer-reviewed articles'},
]
df = pd.DataFrame(contributions)
# Display the table using st.table
st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

'## References'
st.markdown('[1] Jie and Han. [A music recommendation algorithm based on clustering and latent factor model.](https://www.matec-conferences.org/articles/matecconf/pdf/2020/05/matecconf_cscns2020_03009.pdf)')
st.markdown('[2] Pérez-Marcos and López. [Recommender System Based on Collaborative Filtering for Spotify"s Users](https://www.researchgate.net/publication/318511102_Recommender_System_Based_on_Collaborative_Filtering_for_Spotify"s_Users)')
st.markdown('[3] Schedl, Zamani, Chen, Deldjoo, and Elahi. [Current Challenges and Visions in Music Recommender Systems Research](https://browse.arxiv.org/pdf/1710.03208.pdf)')
st.markdown('[4] Samoshyn. [Dataset of songs in Spotify](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data)')


