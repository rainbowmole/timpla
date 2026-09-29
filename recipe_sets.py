from recipe_scrapers import scrape_me

scraper = scrape_me("https://www.panlasangpinoy.com/some-adobo-recipe/", wild_mode=True)
print(scraper.title())
print(scraper.ingredients())
print(scraper.instructions())