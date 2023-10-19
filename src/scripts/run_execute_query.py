import os
import sys
from dotenv import load_dotenv


sys.path.append(os.getcwd())
from src.query.query_sparql import execute_sparql
from src.database.database_utils import query_db
import json
import pandas as pd
 

def main():
    # execute_json_queries()
    # execute_all_csv_queries()
    query_df = pd.read_csv('data/queries/cordis/cordis.csv')
    # print(query_df.columns)
    for i in range(6):
        print(f'QUERY#{i}')
        execute_specific_queries(query_df.iloc[i]['ground_truth'], query_df.iloc[i]['predicted_sparql'])
        print(100 * '*')

def execute_json_queries():
    with open('data/queries/cordis/cordis-train.json') as json_file:
        queries = json.load(json_file)

    # query = queries[99]

    for query in queries:
        res = execute_sparql(query=query['sparql_query'], db_id='cordis')
        print(query['sparql_query'])
        print(res)
        print(50 * '_')


def execute_all_csv_queries():
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


def execute_specific_queries(sql_query, sparql_query):
    res_sparql, execution_time = execute_sparql(query=sparql_query, db_id='cordis_temporary')
    res_sql = query_db(f'EXPLAIN ANALYZE {sql_query}', db_name='cordis_temporary')

    print(sparql_query)
    # print(res_sparql)
    print(execution_time)
    print(50 * '_')
    print(sql_query)
    print(res_sql)


if __name__ == "__main__":
    load_dotenv()
    main()