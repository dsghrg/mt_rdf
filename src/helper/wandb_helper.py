import os

import wandb


def initialize_wandb(args, job_type='train'):
    wandb.init(
        # set the wandb project where this run will be logged
        project="mtrdf",
        entity="gehd",
        job_type=job_type,
        group=args.dataset_name,
        # track hyperparameters and run metadata
        config=args.__dict__,
        tags=[
            f"user__gehd",
            f"host__{os.uname()[1]}",
            f"dataset__{args.dataset_name}",
            f"model__{args.model_name}",
            f"seed__{args.seed}",
            f"model_seed__{args.model_seed}"
        ]

    )