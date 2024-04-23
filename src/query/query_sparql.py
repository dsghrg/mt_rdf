import time
from abc import ABC, abstractmethod
from functools import lru_cache

from SPARQLWrapper import JSON, SPARQLWrapper

# # @lru_cache(maxsize=1000)
# def execute_sparql(query, db_id):
#     """This function executes the sparql query and returns the results"""
#     sparql = SPARQLWrapper("http://160.85.252.68:7200/repositories/"+db_id.lower())
#     sparql.setTimeout(60)
#     sparql.setReturnFormat(JSON)
#     sparql.setQuery('PREFIX :  <http://unics.cloud/ontology/>\nPREFIX onto: <http://www.ontotext.com/>' + query)
#     # the previous query as a literal string
#     try:
#         start_time = time.time()
#         results = sparql.query()
#         execution_time = time.time() - start_time

#         results = results.convert()
#         clean_results = []
#         for row in results['results']['bindings']:
#             interim_results=[]
#             for i in range(len(results['head']['vars'])):
#                 if 'datatype' in row.get(results['head']['vars'][i]).keys():
#                     value = int(row.get(results['head']['vars'][i])['value'])
#                     interim_results.append(value)
#                 else:
#                     value = row.get(results['head']['vars'][i])['value'] 
#                     interim_results.append(value)
#             if len(interim_results) != 0:
#                 clean_results.append(tuple(interim_results))
#         return clean_results, execution_time
#     except Exception as e:
#         print(e)
#     return None, 0

class SparqlQuery(ABC):
    def __init__(self, endpoint="http://172.17.0.1:9999/sparql"):
        self.endpoint = endpoint

    @abstractmethod
    def turn_off_optimization(self, query: str):
        pass

    def execute_sparql(self, query, timeout=600, no_results=True, force_order=False):
        """This function executes the sparql query and returns the results"""
        if force_order:
            query = self.turn_off_optimization(query)

        sparql = SPARQLWrapper(self.endpoint)
        sparql.setTimeout(timeout)
        sparql.setReturnFormat(JSON)
        # sparql.setQuery('PREFIX :  <http://unics.cloud/ontology/>\nPREFIX onto: <http://www.ontotext.com/>' + query)
        sparql.setQuery(query)
        # the previous query as a literal string
        try:
            start_time = time.time()
            results = sparql.query()
            execution_time = time.time() - start_time

            results = results.convert() if not no_results else None

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
            return str(e), 0
        
class Blazegraph(SparqlQuery):
    def __init__(self, endpoint="http://172.17.0.1:9999/sparql"):
        super().__init__(endpoint=endpoint)

    def turn_off_optimization(self, query: str):
        select, where = query.split('WHERE {')
        opt_off = 'hint:Query hint:optimizer "None".'
        query = f'{select} WHERE {{\n\t{opt_off} {where}'

        return query

class Virtuoso(SparqlQuery):
    def __init__(self, endpoint="http://160.85.252.245:8890/sparql"):
        super().__init__(endpoint=endpoint)
            
    def turn_off_optimization(self, query: str):
        query = f'DEFINE sql:select-option "order" {query}'

        return query
