import argparse
import json

from src.helper.logging_helper import *
from src.helper.path_helper import *

setup_logging()


DEFAULT_SEED = 13
DEFAULT_MODEL_SEED = 13
DEFAULT_SEQ_LENGTH = 512
DEFAULT_TRAIN_FRAC = 0.8


class Config():
    DATASET = {
        'ppaths_join': 'ppaths_join.csv',
        'ppaths_join_nl_sparql': 'ppaths_join_nl_sparql.csv',
        'ppaths_join_nl': 'ppaths_join_nl.csv',
        'ppaths_join_nl_encoded': 'ppaths_join_nl_encoded.csv',
    }

def read_arguments_train():
    parser = argparse.ArgumentParser(description='Run training with following arguments')

    # TODO: add default
    parser.add_argument('--dataset_name', help='Choose a dataset to be processed.')
    parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str)
    parser.add_argument('--model_name', type=str)

    parser.add_argument('--experiment_name', type=str, default=None)

    parser.add_argument('--is_encoded', action='store_true', default=True, help='Whether the dataset is already encoded or not.')
    parser.add_argument('--not_encoded', dest='is_encoded', action='store_false')
    
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_seq_length', default=DEFAULT_SEQ_LENGTH, type=int)
    # parser.add_argument('--do_lower_case', action='store_true', default=True)
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int)
    parser.add_argument('--model_seed', default=DEFAULT_MODEL_SEED, type=int)
    parser.add_argument('--use_validation_set', action='store_true')

    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--use_softmax_layer', action='store_true', default=False)

    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_config', action='store_true')
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--no_wandb', dest='wandb', action='store_false')

    args = parser.parse_args()

    if args.save_model:
        args.save_config = True
    
    return args

def write_config_to_file(args):
    config_path = experiment_config_path(args.experiment_name)
    file_exists_or_create(config_path)

    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logging.info(f'\tSuccessfully saved configuration at {config_path}')