import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.helper.logging_helper import setup_logging
from src.helper.wandb_helper import initialize_wandb
from src.models.config import *
from src.models.lstm_attention import LSTMWithAttention


def main(args):
    input_size, hidden_size, output_size = 4096, 150, 1
    args.num_layers = 4
    args.bidirectional = False
    model = LSTMWithAttention(input_size, hidden_size, output_size, args)
    model.to(model.device)
    model.train_model()

if __name__ == "__main__":
    setup_logging()
    load_dotenv()
    args = read_arguments_train()

    if args.wandb:
        initialize_wandb(args)
    main(args)