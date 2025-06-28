# to run 
# scrapy crawl tmdb_spider -o movies.csv -a subdir=671-harry-potter-and-the-philosopher-s-stone

import scrapy

class TmdbSpider(scrapy.Spider):
    name = 'tmdb_spider'
    def __init__(self, subdir="", *args, **kwargs):
        self.start_urls = [f"https://www.themoviedb.org/movie/{subdir}/"]


    # parse the movie page and navigate to the cast page
    def parse(self, response):
        """
        parse the movie page and navigate to the full cast & crew page
        arguments: the response object containing the movie page HTML
        yields a request to follow the cast page URL, with parse_full_credits as callback
        """
        cast_page_link = response.css('.new_button a::attr(href)').get()
        yield response.follow(cast_page_link, callback = self.parse_full_credits) # follow URL of the cast page


    # parse the cast page and navigate to individual actors
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
    
    # parse the actor page
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
                titles = all_titles[i].css('table.credit_group tr a.tooltip bdi::text').getall()

                # get unique titles using set()
                unique_titles = list(set(titles))

                for title in unique_titles:
                    yield {"actor": actor_name, "movie_or_TV_name": title}
                break # break when category is no longer "acting"


