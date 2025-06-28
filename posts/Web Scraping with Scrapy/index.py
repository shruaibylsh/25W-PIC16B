#!/usr/bin/env python
# coding: utf-8
---
title: "Create A Movie Recommender through Web Scraping"
author: "Sihui Lin"
date: "2025-02-10"
categories: [homework, tutorials]
---
# # Create a Movie Recommender through Web Scraping
# In this tutorial, we'll explore how to build a movie recommendation system using web scraping with Python's Scrapy framework. While there are various established approaches to building recommender systems (such as content-based filtering and collaborative filtering), we'll take an innovative approach by leveraging existing movie databases online. Our method will focus on finding movies that share the most actors with a target movie, using this cast overlap as a basis for generating relevant recommendations.

# ## Set Up the Spider
# For this tutorial, we will use TMDB (https://www.themoviedb.org/) to access comprehensive movie data, including cast and crew information.

# First, let's create a file in the spiders directory called `tmdb_spider.py`. We will add the following lines to the file:

# In[1]:


import scrapy

class TmdbSpider(scrapy.Spider):
    name = 'tmdb_spider'
    def __init__(self, subdir="", *args, **kwargs):
        self.start_urls = [f"https://www.themoviedb.org/movie/{subdir}/"]


# This creates a spider named `tmdb_spider` that will start scraping from a specified TMDB movie page. The subdir parameter allows us to specify which movie's page to begin scraping.

# This spider follows a three-stage scraping process. It begins at a movie's main page, and then navigates to the Full Cast & Crew page. Once it reaches the cast page, it will access the individual actor pages. For each actor, it then collects a list of other movies or TV shows they have appeared in. 
# 
# We're going to create three parsing methods for each stage of the scraping.

# ### Parsing the Movie Page
# The first parsing method we write in the spider will assume that we start on a movie page, and then navigate to the Full Cast & Crew Page. We write it as the follows:

# In[2]:


def parse(self, response):
        """
        parse the movie page and navigate to the full cast & crew page
        arguments: the response object containing the movie page HTML
        yields a request to follow the cast page URL, with parse_full_credits as callback
        """
        
        cast_page_link = response.css('.new_button a::attr(href)').get()
        yield response.follow(cast_page_link, callback = self.parse_full_credits) # follow URL of the cast page


# This method appends "/cast" to the movie page URL to access the Cast & Crew page. When called, it yields a new request that Scrapy will follow, using `parse_full_credits` as the callback method to process the resulting page.

# ### Parsing the Full Cast & Crew Page
# The second parsing method, `parse_full_credits`, will access the individual pages of actors (not including crew members) in the specified movie.

# In[3]:


def parse_full_credits(self, response):
    """
    parses the full cast and crew page and navigate to actors' individual pages
    arguments: the response object containing the cast & crew page HTML
    yields: request to follow each actors' page URL, with parse_actor_page as callback
    """
    
    # access the relative URLs for actors' individual pages
    actor_pages = response.css('ol.people.credits:not(.crew) li > a::attr(href)').getall()

    # yield new request for each actor page
    for actor in actor_pages:
        yield response.follow(actor, callback = self.parse_actor_page)


# In this method, we first use a CSS selector to find all the relative URLs for actors' individual pages. We use the symbol '>' to make sure that we only select `a` elements that are direct children of `li` elements. It then iterates through each of these URLs and create a new request for each actor's page, which will then be processed by the `parse_actor_page` method that we will look into next.

# ### Parsing the Actor Page
# Now we have reached the third stage of scraping! We will write the third parsing method, `parse_actor_page`. This method collects each actor's name and their complete list of acting roles, generating a dictionary for each unique movie or TV show they've appeared in. 

# In[4]:


def parse_actor_page(self, response):
    """
    parse the actor page and return a dictionary for actor name and movie/tv names
    argument: the response object containing the actor's page HTML
    yield a dictionary of the form {"actor" : actor_name, "movie_or_TV_name" : movie_or_TV_name},
    which records a list of unique movie/tv names where the actor has acted
    """
    # extract actor name
    actor_name = response.css('h2.title a::text').get()

    # extract the categories the actor have worked in (acting, production, ...)
    categories = response.css('div.credits_list h3::text').getall()

    # extract the table of credits for all categories
    all_titles = response.css('div.credits_list table.credits')

    for i, category in enumerate(categories):
        if category == 'Acting': # only check the acting categories
            titles = all_titles[i].css('table.credit_group tr a.tool:tip bdi::text').getall()

            # get unique titles using set()
            unique_titles = list(set(titles))

            for title in unique_titles:
                yield {"actor": actor_name, "movie_or_TV_name": title}
            break # break when category is no longer "acting"


# This method first extracts the actor's name, the categories the actor have worked in (acting, production...), and the table of credits for all categories. It then checks if the category is "Acting", and then extract the titles of movies/TV under the acting credits. The function set() is used to get unique titles, and from there, we will yield a dictionary containing both the actor's name and the movie/TV title.
# 
# This method will iterate through all actors of the specified movie, so we will get a long list of all the movie/TV titles that each of the actors of the movie have acted in.

# ## Using the Scraper for Movie Recommendation
# One of my favorite movies is Mulholland Drive (2001) directed by David Lynch. It is a masterpiece of surrealist cinema that blends mystery, psychological thriller, and noir elements to create an enigmatic narrative about dreams and reality in Hollywood.
# 
# The link to this movie page is https://www.themoviedb.org/movie/1018-mulholland-drive, and let's test our scraper to see if it works well in generating movie recommendations!
# 
# We will run the following command in the terminal (make sure you are in the same directory as the spider):
# ```bash
# scrapy crawl tmdb_spider -o results.csv -a subdir=1018-mulholland-drive
# ```
# 
# This command will generate a CSV file named `results.csv` in your directory. The file will contain a comprehensive list of actors from Mulholland Drive and their corresponding filmographies, which we can analyze to identify potential movie recommendations based on cast overlap.

# ### Evaluate the Effectiveness 
# Let's evaluate our scraper's effectiveness in generating movie recommendations by analyzing the cast connections.
# 
# First, let's compute a sorted list with the top movies and TC shows that share actors with Mulholland Drive. It will have two columns: "move names" and "number of shared actors".
# 
# Let's first import necessary libraries and load the data:

# In[5]:


import pandas as pd
movie = pd.read_csv("TMDB_scraper/TMDB_scraper/spiders/results.csv")


# Next, we will group the movie dataframe by movie name and count the number of unique actors in each of them. We will then sort the dataframe by the number of shared actors in descending order:

# In[6]:


# Filter out Mulholland Drive to only get other movies
other_movies = movie[~movie['movie_or_TV_name'].isin(['Mulholland Dr.', 'Mulholland Drive'])]

# Group by movie name and count unique actors
movie_connections = other_movies.groupby('movie_or_TV_name')['actor'].nunique().reset_index()

# Rename columns to match desired format
movie_connections.columns = ['movie names', 'number of shared actors']

# Sort by number of shared actors in descending order
movie_recommendations = movie_connections.sort_values(by='number of shared actors', ascending=False)

movie_recommendations


# ### Visualizing the Result of Movie Recommendations
# Let's visualize the result we get above using a bar plot. For this visualization, we will look at the top 10 movies recommended by the scraper.

# In[7]:


# import necessary libraries
import plotly.io as pio
pio.renderers.default="iframe"


# In[9]:


from plotly import express as px

fig = px.bar(movie_recommendations.head(10),
            x='number of shared actors',
            y='movie names',
            orientation='h',
            title='Top 10 Movies/TV Shows Recommended by the Scraper')

fig.show()


# Great! Do these recommended movies meet your expectations? Or will there be better ways for us to generate better movie recommendations?
