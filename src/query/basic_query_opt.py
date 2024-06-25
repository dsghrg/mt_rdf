import logging
import os
import re
import sys
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from src.helper.logging_helper import *

setup_logging()


class QueryOpt(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(QueryOpt, cls).__new__(cls)
        return cls.instance
    
    def __init__(self) -> None:
        base_path = './'
        self.subject_counts, self.predicate_counts, self.object_counts = self._load_dicts(base_path)
    
    def _load_dicts(self, base_path: str) -> tuple[dict, dict, dict]:
        logging.info('Loading subject counts from file...')
        subject_counts = self._load_counts_from_file(os.path.join(base_path, 'subject_counts.txt'))

        logging.info('Loading predicate counts from file...')
        predicate_counts = self._load_counts_from_file(os.path.join(base_path, 'predicate_counts.txt'))

        logging.info('Loading object counts from file...')
        object_counts = self._load_counts_from_file(os.path.join(base_path, 'object_counts.txt'))

        return subject_counts, predicate_counts, object_counts
    
    def _load_counts_from_file(self, file_path: str) -> dict:
        counts_dict = {}
        
        # Get the file size for progress bar
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'r') as file:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading Counts") as pbar:
                for line in file:
                    item, count = line.strip().split('\t')
                    counts_dict[item] = int(count)
                    pbar.update(len(line.encode('utf-8')))
                    
        return counts_dict


    def optimize_query(self, query: str) -> str:
        query = query.replace('\n', ' ')
        triples = self.get_triples_from_sparql(query)
        
        big_n = 100_000_000_000
        variable_counts = defaultdict(lambda: big_n)
        counts = []
        for triple in triples:
            subject, predicate, obj = triple
            predicate = re.sub(r'[\+\*?]+$', '', predicate)
            subject_count = self.subject_counts[subject] if not subject.startswith('?') else variable_counts[subject]
            predicate_count = self.predicate_counts[predicate] if not predicate.startswith('?') else variable_counts[predicate]
            object_count = self.object_counts[obj] if not obj.startswith('?') else variable_counts[obj]
            counts.append((subject_count, predicate_count, object_count, triple))
        
        sorted_triples = []
        # sort loop:
        # 1. find the triple with the smallest counts
        # 2. update the counts: if a variable was already resolved in a previous triples, set its count to 0
        # 3. recalculate the counts for the remaining triples
        while counts:
            # find smallest triple
            counts.sort(key=lambda x: sum(x[0:3]))
            smallest = counts.pop(0)
            sorted_triples.append(smallest[3])
            
            # update counts of variables
            subject, predicate, obj = smallest[3]
            if subject.startswith('?'):
                variable_counts[subject] = 0
            if predicate.startswith('?'):
                variable_counts[predicate] = 0
            if obj.startswith('?'):
                variable_counts[obj] = 0
            
            # recalc for remaining triples
            new_counts = []
            for subject, predicate, obj in [t[3] for t in counts]:
                clean_pred = re.sub(r'[\+\*?]+$', '', predicate)
                subject_count = self.subject_counts[subject] if not subject.startswith('?') else variable_counts[subject]
                predicate_count = self.predicate_counts[clean_pred] if not predicate.startswith('?') else variable_counts[predicate]
                object_count = self.object_counts[obj] if not obj.startswith('?') else variable_counts[obj]
                new_counts.append((subject_count, predicate_count, object_count, (subject, predicate, obj)))
            counts = new_counts

        optimized_query = self.convert_triples_to_query(sorted_triples)
        
        return optimized_query


            # rules for optimizing the query

    def convert_triples_to_query(self, triples: list[tuple[str, str, str]]) -> str:
        query_body = ' .\n  '.join(f'{sub} {pred} {obj}' for sub, pred, obj in triples) + ' .'
        return f'SELECT * WHERE {{\n  {query_body}\n}}'

    def get_triples_from_sparql(self, query: str) -> list[tuple[str, str, str]]:
        # Remove the part before the WHERE clause
        query_body = re.search(r'WHERE\s*\{(.*)\}', query, re.DOTALL).group(1).strip()
        
        # Regular expression to match triples
        triple_pattern = re.compile(r'(\S+)\s+(\S+)\s+(\S+)\s*\.')

        # Find all triples in the query body
        triples = triple_pattern.findall(query_body)

        parsed_triples = []

        for triple in triples:
            subject, predicate, obj = triple

            # # If the subject, predicate, or object is a variable, save None
            # subject = None if subject.startswith('?') else subject
            # predicate = None if predicate.startswith('?') else predicate
            # obj = None if obj.startswith('?') else obj

            parsed_triples.append((subject, predicate, obj))

        return parsed_triples
    
