import streamlit as st
import pandas as pd
'### Spotify Recommendation System' 

'## Introduction/Background:' #A quick introduction of your topic and mostly literature review of what has been done in this area. 
'Finding new music is something that everyone struggles with. We aim to quell the struggle using machine learning to suggest songs similar to user taste, by analyzing existent songs on Spotify according to their features of danceability, energy, and others.'
'By doing this, we will be able to recommend songs to users according to several filters that they select, which will be according to mood, vocals, genre, and others.'
'To accomplish this, we plan on using a dataset containing information about thousands of songs on spotify of various genres, in order to find similar traits amongst certain songs'
'We will be using a [Kaggle dataset](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data) of Spotify songs and playlists which has data on each song''s metrics according to its danceability, energy, speechiness, instrumentalness, and more.'
'## Problem definition: '
'Many people enjoy listening to music, however they get bored of listening to the same songs over and over again. We aim to solve this problem by recommending songs to users based on their preferences.'
'Most people do not know how to find new music, they only know what mood or genre they want to listen to. This project fixes that need by simply asking the user for certain baseline preferences, and recommending the right music accordingly.'
'## Methods:' # What algorithms or methods are you going to use to solve the problems. (Note: Use existing packages/libraries)'
'We are going to use scikit learn to implement a bottom-up agglomerative hierarchical clustering algorithm to cluster the songs into groups based on their features. We will then use the clusters to recommend songs to users based on their preferences.'
'First, we will cluster them by individual features likes danceability and energy, gradually building up the hierarchy until at the top level, we cluster them by genre.'
'Potential results and Discussion:' #Discuss about what type of quantitative metrics your team plan to use for the project (i.e. ML Metrics).
'Since we know the true cluster assignments of the songs based on their data and genres, we will use the scikit.metrics module to implement Rand Index as our evaluation metric.'
'This ignores permutations and requires knowledge of ground truth classes, which we have in this case, given the song data and genres.'

st.markdown(':red[TODO:'
'At At least three references (preferably peer reviewed). You need to properly cite the references on your proposal. This part does NOT count towards word limit.'
'Add proposed timeline from start to finish and list each project members’ responsibilities. Fall and Spring semester sample Gantt Chart. This part does NOT count towards word limit.'
'A contribution table with all group members’ names that explicitly provides the contribution of each member in preparing the project task. This part does NOT count towards word limit.]')
 # 'A checkpoint to make sure you are working on a proper machine learning related project. You are required to have your dataset ready when you submit your proposal. You can change dataset later. However, you are required to provide some reasonings why you need to change the dataset (i.e. dataset is not large enough because it does not provide us a good accuracy comparing to other dataset; we provided accuracy comparison between these two datasets). The reasonings can be added as a section to your future project reports such as midterm report.'


'## Contribution Table'

# Create a list of dictionaries with each member's name and their contribution
contributions = [
    {'Name': 'Adhish Rajan', 'Contribution': 'Creating Gantt Chart and identifying methods/algorithms'},
    {'Name': 'Michael Osmolovskiy', 'Contribution': 'Creating Github repository, finding dataset, and writing proposal'},
    {'Name': 'Abhinav', 'Contribution': 'Creating presentation slides for video and recording video proposal'},
    {'Name': 'Arin Khanna', 'Contribution': 'Creating Gantt Chart'},
    {'Name': 'Vedesh Yadlapalli', 'Contribution': 'Researching and identifying the problem being solved, writing proposal'},
]
df = pd.DataFrame(contributions)
# Display the table using st.table
st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)


