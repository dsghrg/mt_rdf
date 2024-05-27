import os
import re
import sys
from functools import reduce

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.query.query_type import QueryType
from src.query.wdbench import WDBench


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
    base_path = 'data/queries/wdbench/full_wdbench/'

    wdbench = WDBench()
    for index, row in tqdm(wdbench.queries.iterrows(), total=wdbench.queries.shape[0]):
        if not (row['q_type'] == 'multiple_bgps'):
            continue
        sparql_query = wdbench.get_sparql_from_id_and_type(q_id=row['q_id'], q_type=QueryType.get(row['q_type']))
        urls = extract_urls_from_sparql(sparql_query)
        page_titles = get_page_titles(urls)

        new_query = reduce(lambda a, kv: a.replace(*kv), page_titles.items(), sparql_query)
        with open(f'{base_path}/{row["q_type"]}/Q{row["q_id"]}.sparql', 'w') as f:
            f.write(new_query)





if __name__ == "__main__":
    main()
