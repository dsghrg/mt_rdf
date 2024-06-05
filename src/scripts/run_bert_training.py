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
from src.models.pytorch_model import PyTorchModel

setup_logging()



def main(args):
    model = PyTorchModel(args)

    model.train()

if __name__ == "__main__":
    load_dotenv()
    args = read_arguments_train()

    if args.wandb:
        initialize_wandb(args)
    main(args)