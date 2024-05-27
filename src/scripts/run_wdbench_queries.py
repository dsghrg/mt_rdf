import argparse
import logging
import os
import sys
import time

import pandas as pd
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.query.query_sparql import *
from src.query.query_type import QueryType
from src.query.wdbench import WDBench

res_path = 'results/wdbench/'

LIMIT = 300_000


def read_arguments():
    parser = argparse.ArgumentParser(description='Test model with following arguments')
    parser.add_argument('--blazegraph', action='store_true', default=True)
    parser.add_argument('--virtuoso', dest='blazegraph', action='store_false')
    parser.add_argument('--forced', action='store_true', default=False)
    args = parser.parse_args()

    for argument in vars(args):
        logging.info("argument: {} =\t{}".format(str(argument).ljust(20), getattr(args, argument)))

    return args

def parse_to_sparql(query):
    if not LIMIT:
        return f'SELECT * WHERE {{ {query} }}'
    return f'SELECT * WHERE {{ {query} }} LIMIT {LIMIT}'


def res_to_logs(result: dict, forced):
    res_df = pd.DataFrame(result)
    res_df.to_csv(res_path + f'results_wdbench_all{'_forced' if forced else ''}.csv', index=False)



def main(args):
    if args.blazegraph:
        sparql_query = Blazegraph()
    else:
        sparql_query = Virtuoso()
        
    wdbench = WDBench()
    wdbench.queries['exec_n'] = 0
    wdbench.queries['exec_time'] = 900
    
    res_dict = {'query_id': [], 'q_type': [], 'exec_n': [], 'exec_time': [], 'results': []}
    for qtype in QueryType:
        index = 0

        query_df = wdbench.queries[wdbench.queries['q_type'] == qtype.value]
        for j, row in tqdm(query_df.iterrows(), total=query_df.shape[0]):
            for i in range(4):
                query = parse_to_sparql(row['query_parts'])
                _, exec_time = sparql_query.execute_sparql(query=query, force_order=args.forced, timeout=900)
                res_dict['query_id'].append(row['q_id'])
                res_dict['q_type'].append(qtype.value)
                res_dict['exec_n'].append(i)
                res_dict['exec_time'].append(exec_time)

            if index % 10 == 0:
                res_to_logs(result=res_dict, query_type=qtype.value)

            index += 1
        res_to_logs(res_dict)


if __name__ == "__main__":
    args = read_arguments()
    main(args)
