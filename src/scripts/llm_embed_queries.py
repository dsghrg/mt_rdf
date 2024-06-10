import os
import sys

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.getcwd())
import pandas as pd

import src.models.config as config
from src.helper.logging_helper import setup_logging
from src.helper.seed_helper import initialize_gpu_seed

setup_logging()
load_dotenv()


ACCESS_TOKEN=os.environ.get('HUGGINGFACE_TOKEN')


def process_in_batches(model, entries, batch_size, query_prefix, max_length):
    encoded_queries = []
    for i in tqdm(range(0, len(entries), batch_size)):
        batch = entries[i:i + batch_size]
        encoded_batch = model.encode(batch, instruction=query_prefix, max_length=max_length)
        encoded_queries.extend(encoded_batch)
    return encoded_queries


def main():
    device, _ = initialize_gpu_seed(config.DEFAULT_MODEL_SEED)

    login(token=ACCESS_TOKEN)
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v1', trust_remote_code=True)
    model.to(device)

    task_name_to_instruct = {"example": "Given a Natural Language Question based on a SPARQL Query, return '1' if the original SPARQL Query has a faster runtime if you turn off the virtuoso optmizer or '0' otherwise.",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    
    # careful, check data first before setting this length
    max_length = 220
    df = pd.read_csv('data/raw/wdbench_nl.csv')


    queries = list(df['query'])

    encoded_queries = process_in_batches(model, queries, 10, query_prefix, max_length)

    df['encoding'] = encoded_queries
    df.to_csv('data/raw/wdbench_nl_encoded.csv', index=False)

    torch.save(encoded_queries, 'data/raw/wdbench_nl_encoding.pt')


if __name__ == "__main__":
    main()
