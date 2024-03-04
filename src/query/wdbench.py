import re

import pandas as pd

import src.helper.path_helper as ph
from src.query.query_type import QueryType


class WDBench(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(WDBench, cls).__new__(cls)
        return cls.instance
    
    def __init__(self) -> None:
        self.queries = self.__load_queries__()
    
    def __load_queries__(self) -> pd.DataFrame:
        df_opts = pd.read_csv(ph.wdbench_query_path(QueryType.opts), header=None)
        df_opts.rename(columns={0: 'id', 1: 'query_parts'}, inplace=True)
        df_opts['q_type'] = QueryType.opts.value

        df_c2rpqs = pd.read_csv(ph.wdbench_query_path(QueryType.c2rpqs), header=None)
        df_c2rpqs.rename(columns={0: 'id', 1: 'query_parts'}, inplace=True)
        df_c2rpqs['q_type'] = QueryType.c2rpqs.value

        df_paths = pd.read_csv(ph.wdbench_query_path(QueryType.paths), header=None)
        df_paths.rename(columns={0: 'id', 1: 'query_parts'}, inplace=True)
        df_paths['q_type'] = QueryType.paths.value

        df_single_bgps = pd.read_csv(ph.wdbench_query_path(QueryType.single_bgps), header=None)
        df_single_bgps.rename(columns={0: 'id', 1: 'query_parts'}, inplace=True)
        df_single_bgps['q_type'] = QueryType.single_bgps.value

        df_multiple_bgps = pd.read_csv(ph.wdbench_query_path(QueryType.multiple_bgps), header=None)
        df_multiple_bgps.rename(columns={0: 'id', 1: 'query_parts'}, inplace=True)
        df_multiple_bgps['q_type'] = QueryType.multiple_bgps.value

        df_queries = pd.concat([df_opts, df_c2rpqs, df_paths, df_single_bgps, df_multiple_bgps], ignore_index=True)
        df_queries.rename(columns={'id': 'q_id'}, inplace=True)

        df_queries = self.__get_query_stats__(df_queries)

        # query_stat_dict = {qtype.value: {} for qtype in QueryType}

        # query_stat_dict = self.__get_query_stats__(query_stat_dict=query_stat_dict, df=df_opts, qtype=QueryType.opts.value)
        # query_stat_dict = self.__get_query_stats__(query_stat_dict=query_stat_dict, df=df_c2rpqs, qtype=QueryType.c2rpqs.value)
        # query_stat_dict = self.__get_query_stats__(query_stat_dict=query_stat_dict, df=df_multiple_bgps, qtype=QueryType.multiple_bgps.value)
        # query_stat_dict = self.__get_query_stats__(query_stat_dict=query_stat_dict, df=df_paths, qtype=QueryType.paths.value)
        # query_stat_dict = self.__get_query_stats__(query_stat_dict=query_stat_dict, df=df_single_bgps, qtype=QueryType.single_bgps.value)

        # dict_df = {'q_type': [], 'q_id': [], 'n_entities': [], 'entities': []}
        # for k, v in query_stat_dict.items():
        #     for id_key, entity_list in v.items():
        #         dict_df['q_type'].append(k)
        #         dict_df['q_id'].append(id_key)
        #         dict_df['n_entities'].append(len(entity_list))
        #         dict_df['entities'].append(entity_list)

        # query_df = pd.DataFrame(data=dict_df)

        return df_queries

    def __get_query_stats__(self, df):
        entity_pattern = r'<http://www\.wikidata\.org/entity/.+?>'
        df['n_entities'] = 0
        df['entities'] = ''
        for index, row in df.iterrows():
            entities = re.findall(entity_pattern, row['query_parts'])
            df.at[index, 'n_entities'] = len(entities)
            df.at[index, 'entities'] = entities
        
        return df
    
    def parse_to_sparql(self, query, limit: int=None):
        if not limit:
            return f'SELECT * WHERE {{ {query} }}'
        return f'SELECT * WHERE {{ {query} }} LIMIT {limit}'
    
    def get_sparql_from_id_and_type(self, q_id: int, q_type: QueryType) -> str:
        return self.parse_to_sparql(self.queries[(self.queries['q_id'] == q_id) & (self.queries['q_type'] == q_type.value)]['query_parts'].item())
