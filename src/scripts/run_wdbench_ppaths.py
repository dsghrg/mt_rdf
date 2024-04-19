import os
import sys
import time

import pandas as pd
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.query.query_sparql import SparqlQuery

res_path = 'results/wdbench/'

LIMIT = 300_000



def main():
    sparql_query = SparqlQuery()
    dir_path = 'data/queries/wdbench/ppaths/original/'

    res_dict = {'query_id': [], 'exec_time': [], 'results': []}
    index = 0
    for filename in os.listdir(dir_path):
        with open(os.path.join(dir_path, filename), 'r') as file:
            query = file.read()
            res, exec_time = sparql_query.execute_sparql(query)
            filename = filename.split('.')[0]
            res_dict['query_id'].append(filename)
            res_dict['exec_time'].append(exec_time)
            res_dict['results'].append(res)

            if index % 10 == 0:
                res_df = pd.DataFrame(res_dict)
                res_df.to_csv(res_path + f'ppaths/results_ppaths.csv')
        index += 1
if __name__ == "__main__":
    main()
