import os
import sys

sys.path.append(os.getcwd())
from src.query.query_sparql import execute_sparql
import json
import pandas as pd
 

def main():
    # with open('data/queries/cordis/cordis-train.json') as json_file:
    #     queries = json.load(json_file)
    query_df = pd.read_csv('data/queries/cordis/cordis.csv')
    for i, row in query_df.iterrows():

        # query = queries[99]
        res = execute_sparql(query=row['predicted_sparql'], db_id='cordis_temporary')
        print(row['predicted_sparql'])
        print(res)
        print(50*'_')

if __name__ == "__main__":
    main()