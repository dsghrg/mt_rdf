import os
import re
from functools import reduce

import requests
from bs4 import BeautifulSoup


def extract_urls_from_sparql(sparql_query):
    # Regular expression to match URLs
    url_pattern = r'<(http://www\.wikidata\.org/[^>]+)>'
    
    # Find all matches of URLs in the SPARQL query
    urls = re.findall(url_pattern, sparql_query)
    
    return urls


def get_page_titles(urls):
    page_titles = {}
    for url in urls:
        search_url = url.replace('entity', 'wiki')
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        title_span = soup.find('span', class_='wikibase-title-label')
        if title_span:
            page_titles[url] = title_span.text.strip()
        else:
            page_titles[url] = "Title not found"
            print(url)
    return page_titles

def main():
    base_path = 'data/queries/wdbench/ppaths/'

    for filename in os.listdir(f"{base_path}original/"):
        with open(f"{base_path}original/{filename}", 'r') as file:
            sparql_query = file.read()
            urls = extract_urls_from_sparql(sparql_query)
            page_titles = get_page_titles(urls)

            new_query = reduce(lambda a, kv: a.replace(*kv), page_titles.items(), sparql_query)
            with open(f'{base_path}nl/{filename}', 'w') as f:
                f.write(new_query)


    print('res')



if __name__ == "__main__":
    main()
