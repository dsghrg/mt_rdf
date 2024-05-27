import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.helper.logging_helper import setup_logging
from src.query.query_sparql import *
from src.query.query_type import QueryType

setup_logging()

base_data_path = 'data/queries/wdbench/full_wdbench/'

def read_arguments_matching():
    parser = argparse.ArgumentParser(description='Test model with following arguments')
    # parser.add_argument('--blazegraph', action='store_true', default=True)
    # parser.add_argument('--virtuoso', dest='blazegraph', action='store_false')
    # parser.add_argument('--query_mode', default='original', choices=['original', 'opt_blaze', 'opt_virt'])
    # parser.add_argument('--forced', action='store_true', default=False)
    args = parser.parse_args()

    for argument in vars(args):
        logging.info('argument: {} =\t{}'.format(str(argument).ljust(20), getattr(args, argument)))

    return args

def main(args):
    client = OpenAI(
        api_key=os.environ.get('OPENAI_API_KEY')
    )

    for q_type in QueryType:
        print(q_type)
        for filename in tqdm(os.listdir(f'{base_data_path}{q_type}/')):
            with open(f'{base_data_path}{q_type}/{filename}', 'r+') as file:
                # here, position is initially at the beginning
                sparql_query = file.read()

                chat = client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[{'role': 'user', 
                                'content': f'I have a sparql query which is used for wikidata and I want you to give me the corresponding natural language question, only that question. Here it is: {sparql_query}'}]
                )
                reply = chat.choices[0].message.content
                
                file.write(f'\n\nNL-QUESTION:\n{reply}')


if __name__ == "__main__":
    load_dotenv()
    args = read_arguments_matching()
    main(args)
