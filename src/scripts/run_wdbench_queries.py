import time
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import os
import sys
from enum import Enum

sys.path.append(os.getcwd())


res_path = 'results/wdbench/'

LIMIT = False


def parse_to_sparql(query):
    if not LIMIT:
        return f'SELECT * WHERE {{ {query} }}'
    return f'SELECT * WHERE {{ {query} }} LIMIT {LIMIT}'


def execute_sparql(query):
    """This function executes the sparql query and returns the results"""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setTimeout(300)
    sparql.setReturnFormat(JSON)
    # sparql.setQuery('PREFIX :  <http://unics.cloud/ontology/>\nPREFIX onto: <http://www.ontotext.com/>' + query)
    sparql.setQuery(query)
    # the previous query as a literal string
    try:
        start_time = time.time()
        results = sparql.query()
        execution_time = time.time() - start_time

        results = results.convert()
        return results, execution_time
    #     clean_results = []
    #     for row in results['results']['bindings']:
    #         interim_results=[]
    #         for i in range(len(results['head']['vars'])):
    #             if 'datatype' in row.get(results['head']['vars'][i]).keys():
    #                 value = int(row.get(results['head']['vars'][i])['value'])
    #                 interim_results.append(value)
    #             else:
    #                 value = row.get(results['head']['vars'][i])['value']
    #                 interim_results.append(value)
    #         if len(interim_results) != 0:
    #             clean_results.append(tuple(interim_results))
    #     return clean_results, execution_time
    except Exception as e:
        print(e)
    return None, 0


def res_to_logs(result: dict, query_type):
    res_df = pd.DataFrame(result)
    res_df.to_csv(res_path + f'results_{query_type}.csv')


def run_all_in_df(query_df, query_type):
    res_dict = {'query_id': [], 'exec_time': []}
    for index, row in query_df.iterrows():
        query = parse_to_sparql(row['query_parts'])
        _, time_s = execute_sparql(query=query)
        res_dict['query_id'].append(row['id'])
        res_dict['exec_time'].append(time_s)
        if index % 10 == 0:
            res_to_logs(result=res_dict, query_type=query_type)


class QueryType(str, Enum):
    c2rpqs = 'c2rpqs'
    multiple_bgps = 'multiple_bgps'
    opts = 'opts'
    paths = 'paths'
    single_bgps = 'single_bgps'


def main():
    for qtype in QueryType:
        df = pd.read_csv(
            f'data/queries/wdbench/{qtype.value}.txt', header=None)
        df.rename(columns={0: 'id', 1: 'query_parts'}, inplace=True)
        print(50*'-')
        print(qtype.value)
        print(50*'-')
        run_all_in_df(query_df=df, query_type=qtype.value)

    # c2rpqs_df = pd.read_csv('data/queries/wdbench/c2rpqs.txt')
    # multiple_bgps_df = pd.read_csv('data/queries/wdbench/multiple_bgps.txt')
    # opts_df = pd.read_csv('data/queries/wdbench/opts.txt')
    # paths_df = pd.read_csv('data/queries/wdbench/paths.txt')
    # single_bgps_df = pd.read_csv('data/queries/wdbench/single_bgps.txt')


if __name__ == "__main__":
    main()
