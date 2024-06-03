import logging
import random

import numpy as np
import torch

from src.helper.logging_helper import setup_logging

setup_logging()


def initialize_gpu_seed(seed: int, cpu_only=False):
    device, n_gpu = setup_gpu(cpu_only)

    init_seed_everywhere(seed, n_gpu)

    return device, n_gpu


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info(f'Set seed for random, numpy and torch to {seed}')


def init_seed_everywhere(seed, n_gpu):
    init_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_gpu(cpu_only=False):
    # Setup GPU parameters
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu_only else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.info('=' * 50)
    logging.info('')
    logging.info(f"\t[PyTorch]\tWe are using {str(device).upper()} on {n_gpu} gpu's.")
    logging.info('')
    logging.info('=' * 50)

    return device, n_gpu