import argparse
import logging
import os
import sys
import time

import pandas as pd
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.helper.logging_helper import setup_logging
from src.query.query_sparql import *

setup_logging()


LIMIT = 300_000

def read_arguments_matching():
    parser = argparse.ArgumentParser(description='Test model with following arguments')
    parser.add_argument('--blazegraph', action='store_true', default=True)
    parser.add_argument('--virtuoso', dest='wandb', action='store_false')
    parser.add_argument('--query_mode', default='original', choices=['original', 'opt_blaze', 'opt_virt'])
    parser.add_argument('--forced', action='store_true', default=False)
    args = parser.parse_args()

    for argument in vars(args):
        logging.info("argument: {} =\t{}".format(str(argument).ljust(20), getattr(args, argument)))

    return args

def main(args):
    if args.blazegraph:
        sparql_query = Blazegraph()
    else:
        sparql_query = Virtuoso()


    dir_path = 'data/queries/wdbench/ppaths/'
    if args.query_mode == 'original':
        dir_path += 'original'
    elif args.query_mode == 'opt_blaze':
        if not args.blazegraph:
            raise Exception('Blazegraph optimized queries cannot be run on Virtuoso.')
        dir_path += 'opt_blaze'
    else:
        if args.blazegraph:
            raise Exception('Virtuoso optimized queries cannot be run on Blazegraph.')
        dir_path += 'opt_virt'

    res_dict = {'query_id': [], 'exec_n': [], 'exec_time': [], 'results': []}
    index = 0
    result_path = f'results/wdbench/ppaths/{'blaze' if args.blazegraph else 'virt'}/'
    for filename in tqdm(os.listdir(dir_path)): 
        with open(os.path.join(dir_path, filename), 'r') as file:
            query = file.read()
            for i in range(4):
                res, exec_time = sparql_query.execute_sparql(query, force_order=args.forced, timeout=900)
                filename = filename.split('.')[0]
                res_dict['query_id'].append(filename)
                res_dict['exec_n'].append(i)
                res_dict['exec_time'].append(exec_time)
                res_dict['results'].append(res)

            if index % 10 == 0:
                res_df = pd.DataFrame(res_dict)
                res_df.to_csv(result_path + f'results_ppaths_{args.query_mode}{'_forced' if args.forced else ''}.csv')
        index += 1
if __name__ == "__main__":
    args = read_arguments_matching()
    main(args)
