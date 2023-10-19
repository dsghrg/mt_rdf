import os
import sys

sys.path.append(os.getcwd())
from src.query.query_sparql import execute_sparql
import json
import pandas as pd
 

def main():
    # execute_json_queries()
    execute_csv_queries()

def execute_json_queries():
    with open('data/queries/cordis/cordis-train.json') as json_file:
        queries = json.load(json_file)

    # query = queries[99]

    for query in queries:
        res = execute_sparql(query=query['sparql_query'], db_id='cordis')
        print(query['sparql_query'])
        print(res)
        print(50 * '_')


def execute_csv_queries():
    query_df = pd.read_csv('data/queries/cordis/cordis.csv')
    slow_counter = 0
    for i, row in query_df.iterrows():
        res, execution_time = execute_sparql(query=row['predicted_sparql'], db_id='cordis_temporary')
        print(f'Query#{i+1}')
        print(f'Execution time:\t{execution_time}')
        # print(res)
        print(50*'_')

        if execution_time > 2:
            slow_counter = slow_counter + 1

    print(slow_counter)


if __name__ == "__main__":
    main()