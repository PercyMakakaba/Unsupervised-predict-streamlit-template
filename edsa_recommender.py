"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from importlib.metadata import packages_distributions
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# Data handling dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model



# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
train_data = pd.read_csv("resources/data/train.csv")

#
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background: rgb(58,175,169);
background: linear-gradient(90deg, rgba(58,175,169,1) 35%, rgba(58,175,169,1) 97%, rgba(58,175,169,1) 100%);}}

[data-testid="stSidebar"] > div:first-child {{
background: rgb(23,37,42);
background: linear-gradient(90deg, rgba(23,37,42,1) 35%, rgba(23,37,42,1) 97%, rgba(23,37,42,1) 100%);
}}
"""
st.markdown(page_bg_img, unsafe_allow_html=True)



# App declaration
def main():
    page_options = ["Recommender System","Solution Overview", "Dashboard", "Contact us", 'yes', 'no', 'Contact us']

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        st.write("---")
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
                    

    #Analytics/Dashboard page
    if page_selection == "Dashboard":
        # Load movie data
        movies_data = pd.read_csv("resources/data/movies.csv")

        # Title and description for this section
        st.title("Movie Information")
        st.write("Search for a movie to see details:")

        # Search bar to select a movie
        search_query = st.text_input("Search Movie")
        matching_movies = movies_data[movies_data['title'].str.contains(search_query, case=False)]

        # Check if matching movies found
        if not matching_movies.empty:
            selected_movie = st.selectbox("Select Movie", matching_movies['title'])

            # Display details about the selected movie
            selected_movie_details = movies_data[movies_data['title'] == selected_movie].iloc[0]
            st.write(f"**Title:** {selected_movie_details['title']}")
            st.write(f"**Genres:** {selected_movie_details['genres']}")
            

            # Display ratings information using the train.csv file
            ratings_data = pd.read_csv("resources/data/train.csv")
            movie_ratings = ratings_data[ratings_data['movieId'] == selected_movie_details['movieId']]
            
            if not movie_ratings.empty:
                st.write(f"**Average Rating:** {round(movie_ratings['rating'].mean(),2)}")
            else:
                st.write("**No Ratings Available for this Movie**")
        else:
            st.warning("No matching movies found. Please ")



    #Solution overview page
    if page_selection == "Solution Overview":
        # Load movie and ratings data
        movies_data = pd.read_csv("resources/data/movies.csv")
        ratings_data = pd.read_csv("resources/data/train.csv")


        # Title and description for this section
        st.title("Genre-Based Movie Recommendations")
        st.write("Select a genre to get movie recommendations:")

        # Dropdown to select a genre
        selected_genre = st.selectbox("Select Genre", movies_data['genres'].unique())

        # Get movies of the selected genre
        genre_movies = movies_data[movies_data['genres'].str.contains(selected_genre)]

        # Display a sample of movies of the selected genre
        st.write(f"Sample of {selected_genre} Movies:")
        st.dataframe(genre_movies.head())

        # Recommender: Recommend top-rated movies of the selected genre
        top_genre_movies = genre_movies.merge(ratings_data, on='movieId')
        top_genre_movies = top_genre_movies.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)

        # Display recommended movies
        st.write("Top 10 Recommended Movies in the Selected Genre:")
        st.write(top_genre_movies.index.tolist())

    #Contact us page
    if page_selection == "Contact us":
        st.write("---")

        contact_form = """
        <h4>For more information please contact us...</h4>
        <form action="https://formsubmit.co/your@email.com" method="POST">
        <input type="text" name="name", placeholder="Your name" required>
        <input type="email" name="email", placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
        </form>

        """

        st.markdown(contact_form, unsafe_allow_html=True)

        #use local css file
        def local_css(file_name):
                with open(file_name) as f:
                    st.markdown(f"<style>{f.read()}</styel>", unsafe_allow_html=True)
        
        local_css('style/style.css')
    
        #Another page
    if page_selection == "Another":
        # Load ratings data
        ratings_data = pd.read_csv("resources/data/train.csv")

        # Title and description for this section
        st.title("Top Rated Movies with Posters")
        st.write("Explore the top-rated movies based on user ratings:")

        # Get top-rated movies
        top_rated_movies = ratings_data.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(10)
        top_rated_movie_ids = top_rated_movies.index.tolist()

        # Load movie data to get movie details
        movies_data = pd.read_csv("resources/data/movies.csv")

        # Display details of top-rated movies with posters in a table
        st.write("Top 10 Rated Movies:")
        st.write("")

        # Create a table to display movie details
        table_columns = ["Movie Title", "Genres", "Avg Rating", "Poster"]
        table_data = []

        # Loop through top-rated movies
        for movie_id in top_rated_movie_ids:
            movie_details = movies_data[movies_data['movieId'] == movie_id].iloc[0]
            
            # Get poster URL
            poster_path = f"resources/img/{movie_id}.jpg"
            
            # Add row to the table
            table_data.append([
                f"**{movie_details['title']}**",
                movie_details['genres'],
                f"{top_rated_movies[movie_id]:.2f}",
                f"![Poster]({poster_path})"
            ])

        # Display the table
        st.table(pd.DataFrame(table_data, columns=table_columns).set_index("Movie Title"))


    if page_selection == "yes":

                # Load the data
        tags_data = pd.read_csv("resources/data/tags.csv")
        movies_data = pd.read_csv("resources/data/movies.csv")
        # Function to generate and display word cloud
        def generate_wordcloud():
            # Concatenate all tags into a single string
            all_tags = ' '.join(tags_data['tag'].astype(str))

            # Generate the word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_tags)

            # Display the word cloud using Matplotlib
            st.image(wordcloud.to_image(), use_container_width=True)

        # Function to display a bar chart of movie genres
        def display_genre_bar_chart():
            genre_counts = movies_data['genres'].str.split('|', expand=True).stack().value_counts()
            st.bar_chart(genre_counts)

        # Streamlit app definition
        def main():
            st.title("MovieLens Recommender Dashboard")

            # Add a section for the word cloud
            st.header("User-Generated Tags Word Cloud")
            generate_wordcloud()

            # Add a section for movie genres bar chart
            st.header("Movie Genres Bar Chart")
            display_genre_bar_chart()

            # Add an interactive dropdown for movie genres
            selected_genre = st.selectbox("Select a Genre:", movies_data['genres'].str.split('|').explode().unique())

            # Display movies based on the selected genre
            st.subheader(f"Movies in {selected_genre} genre:")
            movies_in_genre = movies_data[movies_data['genres'].str.contains(selected_genre)]
            st.dataframe(movies_in_genre[['movieId', 'title', 'genres']])

        if __name__ == "__main__":
            main()
                
if __name__ == '__main__':
    main()
