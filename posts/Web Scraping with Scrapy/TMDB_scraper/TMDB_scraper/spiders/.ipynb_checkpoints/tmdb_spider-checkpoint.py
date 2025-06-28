# to run 
# scrapy crawl tmdb_spider -o movies.csv -a subdir=671-harry-potter-and-the-philosopher-s-stone

import scrapy

class TmdbSpider(scrapy.Spider):
    name = 'tmdb_spider'
    def __init__(self, subdir="", *args, **kwargs):
        self.start_urls = [f"https://www.themoviedb.org/movie/{subdir}/"]

    def parse(self, response):
        """Parse movie page and navigate to full credits page.
                Args: Response object containing the movie page HTML
                Yields: Request object for the full credits page
            """
        cast_link = response.css('.new_button a::attr(href)').get()
        yield response.follow(cast_link, callback=self.parse_full_credits) # or we can do this with hard coding:
        # yield response.follow("cast", callback=self.parse_full_credits)


    def parse_full_credits(self, response):
        """Parse the full credits page and navigate to each actor's page.
                Args: Response object containing the full credits page HTML
                Yields: Request objects for each actor's page
            """
            actor_links = response.css('ol.people.credits:not(.crew) li div.info p
        a::attr(href)').getall() for link in actor_links:
        yield response.follow(link, callback=self.parse_actor_page)

    def parse_actor_page(self, response):
        """Parse an actor's page and extract their acting credits.
                Args: Response object containing the actor's page HTML
                Yields: {"actor": str, "movie_or_TV_name": str}
            """
            actor_name = response.css('h2.title a::text').get()
            # Find all section headers and their corresponding tables
            sections = response.css('div.credits_list h3::text').getall()
            tables = response.css('div.credits_list table.credits')
            # Find Acting section and process its table
        for i, section in enumerate(sections): if section.strip() == "Acting":
                    all_rows = tables[i].css('table.credit_group tr')
                    unique_titles = {row.css('a.tooltip bdi::text').get().strip()
        for row in all_rows
        if row.css('a.tooltip bdi::text').get()}
        for title in unique_titles: yield {
                            "actor": actor_name,
                            "movie_or_TV_name": title
                        }
        break
    