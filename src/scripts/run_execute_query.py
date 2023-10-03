import os
import sys

sys.path.append(os.getcwd())
from src.query.query_sparql import execute_sparql
import json
 

def main():
    with open('data/queries/cordis/cordis-train.json') as json_file:
        queries = json.load(json_file)

    for query in queries:

        # query = queries[99]
        res = execute_sparql(query=query['sparql_query'], db_id='cordis')
        print(res)

if __name__ == "__main__":
    main()