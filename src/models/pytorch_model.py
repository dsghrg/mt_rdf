import argparse
import logging
import os

from src.data.dataset import ModelDataset
from src.helper.logging_helper import setup_logging
from src.helper.path_helper import *
from src.helper.seed_helper import initialize_gpu_seed
from src.models.config import write_config_to_file
from src.models.optimizer import build_optimizer

setup_logging()

import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (AutoTokenizer, BertConfig,
                          BertForSequenceClassification)

import wandb


class PyTorchModel:
    def __init__(self, args):
        self.args = args
        self.dataset = ModelDataset(dataset_name=self.args.dataset_name,
                                    seed=self.args.seed,
                                    max_seq_length=self.args.max_seq_length,
                                    do_lower_case=True,
                                    train_frac=0.8,
                                    use_val=self.args.use_validation_set)

        self.seed = self.args.seed
        self.model_seed = self.args.model_seed
        self.use_val = self.args.use_validation_set

        self.network = self._get_pretrained_network(self.args.model_name, self.args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        # Resize the embedding matrix w.r.t the default pretrained model given the new custom tokens
        # self.network.resize_token_embeddings(len(self.dataset.tokenizer))
        # logging.info("Resized the embedding matrix given newly added custom tokens")
        self.device, _ = initialize_gpu_seed(self.model_seed)
        self.network.to(self.device)

        # wandb logs for changes in gradients, weights, biases, activations, etc.
        # if self.args.wandb:
        #     wandb.watch(self.network)

        self._setup_data_loaders()
        self._setup_optimizer()
        self._reset_prediction_buffer()
    
    def save(self, suffix: str = ""):
        file_name = "".join([self.args.model_name, suffix, '.pt'])
        model_path = experiment_file_path(self.args.experiment_name, file_name)

        if file_exists_or_create(model_path):
            raise ValueError(f'Checkpoint already exists at {model_path}')

        torch.save(self.network.state_dict(), model_path)
        logging.info(f"\tSuccessfully saved checkpoint at {model_path}")

        config_path = experiment_config_path(self.args.experiment_name)
        if not os.path.isfile(config_path):
            write_config_to_file(self.args)

    
    def train(self):
        global_step = 0
        total_loss, prev_epoch_loss, prev_loss = 0.0, 0.0, 0.0

        # run zero-shot on test set
        self.test()
        if self.args.save_model:
            self.save(suffix='__epoch0__zeroshot')
        
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc=f'Training for {self.args.num_epochs} epochs ...'):
            sample_count, sample_correct = 0, 0

            for step, data in tqdm(enumerate(self.train_data_loader),
                                          desc=f'[TRAINING] Running epoch {epoch}/{self.args.num_epochs} ...',
                                          total=len(self.train_data_loader)):
                self.network.train()
                ids = data['input_ids'].to(self.device)
                mask = data['mask'].to(self.device)
                labels = data['labels'].to(self.device)
                queries = data['query']
                query_id = data['query_id']

                loss, output = self.network(input_ids=ids, attention_mask=mask, labels=labels).to_tuple()

                # lehl@2022-10-24: Using (Log)SoftmaxLayer in addition with the appropriate loss function
                # (see https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)
                if self.args.use_softmax_layer:
                    # loss_fn = nn.NLLLoss(weight=None, reduction='mean')
                    loss_fn = nn.CrossEntropyLoss(weight=self.dataset.label_weights)
                    loss = loss_fn(output, labels.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)

                total_loss += loss.item()

                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                self.network.zero_grad()

                # Calculate how many were correct
                predictions = torch.argmax(output, axis=1)
                batch_count = len(predictions)
                batch_correct = (predictions == labels.squeeze()).detach().cpu().numpy().sum()

                self._wandb_log({
                    'iteration': global_step,
                    'train_loss__batch': total_loss - prev_loss,
                    'train_acc__batch': batch_correct / batch_count,
                    'lr': self.scheduler.get_last_lr()[0]
                })

                sample_count += batch_count
                sample_correct += batch_correct
                prev_loss = total_loss
                global_step += 1

            train_loss = round((total_loss - prev_epoch_loss) / len(self.train_data_loader), 4)
            train_acc = round(sample_correct / sample_count, 4)
            self._wandb_log({
                'epoch': epoch,
                'iteration': global_step,
                'train_loss': train_loss,
                'train_acc': train_acc
            })
            logging.info(
                f"[Epoch {epoch}/{self.args.num_epochs}]\tTrain Loss: {train_loss}\tTrain Accuracy: {train_acc}")

            prev_epoch_loss = total_loss

            if self.args.save_model:
                self.save(suffix=f'__epoch{epoch}')

            # Run test+val set after each epoch
            self.test(epoch)
            if self.use_val:
                self.validate(epoch)


        

    
    def test(self, epoch: int = 0, global_step: int = 0):
        self._reset_prediction_buffer()
        total_loss, prev_loss = 0.0, 0.0
        sample_count, sample_correct = 0, 0

        for step, data in tqdm(enumerate(self.test_data_loader), desc=f'[TESTING] Running epoch {epoch} ...',
                                      total=len(self.test_data_loader)):
            self.network.eval()
            ids = data['input_ids'].to(self.device)
            mask = data['mask'].to(self.device)
            labels = data['labels'].to(self.device)
            queries = data['query']
            query_id = data['query_id']

            loss, output = self.network(input_ids=ids, attention_mask=mask, labels=labels).to_tuple()
            # import code; code.interact(local=dict(globals(), **locals()))
            total_loss += loss.item()

            # Calculate how many were correct
            predictions = torch.argmax(output, axis=1)

            if self.args.use_softmax_layer:
                prediction_proba = torch.exp(output)[:, 1]
            else:
                prediction_proba = torch.nn.functional.softmax(output, dim=1)[:, 1]

            self.log_predictions(labels, query_id, predictions, prediction_proba, step=step)

            sample_count += len(predictions)
            sample_correct += (predictions == labels.squeeze()).detach().cpu().numpy().sum()

        test_loss = round(total_loss / len(self.test_data_loader), 4)
        test_acc = round(sample_correct / sample_count, 4)

        self._wandb_log({
            'epoch': epoch,
            'iteration': global_step,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_prediction_proba': self._plot_test_prediction_proba_hist(epoch)
        })
        logging.info(f"[Epoch {epoch}/{self.args.num_epochs}]\tTest Loss: {test_loss}\tTest Accuracy: {test_acc}")
        self.save_test_predictions(epoch)

    def validate(self, epoch: int = 0, global_step: int = 0):
        if self.val_data_loader is None:
            logging.info(f'No validation data loader for this dataset -> skipping validation step.')
            return
        self._reset_prediction_buffer(is_test=False)
        total_loss, prev_loss = 0.0, 0.0
        sample_count, sample_correct = 0, 0

        for step, data in tqdm(enumerate(self.val_data_loader), desc=f'[VALIDATE] Running epoch {epoch} ...',
                                      total=len(self.val_data_loader)):
            self.network.eval()
            ids = data['input_ids'].to(self.device)
            mask = data['mask'].to(self.device)
            labels = data['labels'].to(self.device)
            queries = data['query']
            query_ids = data['query_id']

            loss, output = self.network(input_ids=ids, attention_mask=mask, labels=labels).to_tuple()
            total_loss += loss.item()

            # Calculate how many were correct
            predictions = torch.argmax(output, axis=1)

            if self.args.use_softmax_layer:
                prediction_proba = torch.exp(output)[:, 1]
            else:
                prediction_proba = torch.nn.functional.softmax(output, dim=1)[:, 1]

            self.log_predictions(labels, query_ids, predictions, prediction_proba, step=step, is_test=False)

            sample_count += len(predictions)
            sample_correct += (predictions == labels.squeeze()).detach().cpu().numpy().sum()

        val_loss = round(total_loss / len(self.test_data_loader), 4)
        val_acc = round(sample_correct / sample_count, 4)

        self._wandb_log({
            'epoch': epoch,
            'iteration': global_step,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_prediction_proba': self._plot_test_prediction_proba_hist(epoch, is_test=False)
        })
        logging.info(f"[Epoch {epoch}/{self.args.num_epochs}]\tVal Loss: {val_loss}\tVal Accuracy: {val_acc}")

    def _reset_prediction_buffer(self, is_test=True):
        num_test_samples = len(self.test_data_loader.dataset) if is_test else len(self.val_data_loader.dataset)
        self.prediction_buffer = {
            'queries': np.zeros(num_test_samples, dtype=str),
            'labels': np.zeros(num_test_samples, dtype=int),
            'predictions': np.zeros(num_test_samples, dtype=int),
            'prediction_proba': np.zeros(num_test_samples, dtype=float)
        }
    
    def _wandb_log(self, wandb_dict: dict):
        if self.args.wandb:
            wandb.log(wandb_dict)

    def save_test_predictions(self, epoch):
        file_name = "".join([self.args.model_name, '__prediction_log__ep', str(epoch), '.csv'])
        log_path = experiment_file_path(self.args.experiment_name, file_name)

        file_exists_or_create(log_path)

        with open(log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.prediction_buffer.keys())
            writer.writerows(zip(*self.prediction_buffer.values()))

    def log_predictions(self, labels, query_ids, predictions, prediction_proba, step=0, is_test=True):
        def tensor_to_list(tensor_data):
            return tensor_data.detach().cpu().numpy().reshape(-1).tolist()

        batch_size = self.test_data_loader.batch_size
        num_samples = len(self.test_data_loader.dataset) if is_test else len(self.val_data_loader.dataset)

        start_idx = step * batch_size
        end_idx = np.min([((step + 1) * batch_size), num_samples])
        
        # import code; code.interact(local=dict(globals(), **locals()))
        # Save the IDs, labels and predictions in the buffer
        self.prediction_buffer['queries'][start_idx:end_idx] = query_ids
        self.prediction_buffer['labels'][start_idx:end_idx] = tensor_to_list(labels)
        self.prediction_buffer['predictions'][start_idx:end_idx] = tensor_to_list(predictions)
        self.prediction_buffer['prediction_proba'][start_idx:end_idx] = tensor_to_list(prediction_proba)

    def _plot_test_prediction_proba_hist(self, epoch, lower_limit=0, is_test=True):
        if not self.args.wandb:
            return None

        try:
            fig, ax = plt.subplots()

            probs = self.prediction_buffer['prediction_proba']
            probs = probs[probs >= lower_limit]

            cnts, vals, _ = ax.hist(probs, range=[lower_limit, 1], bins=20)
            y_max = 10 ** np.ceil(np.log10(np.max(cnts)))
            ax.set_yscale('log')
            ax.set_ylim([1, y_max])
            ax.set_xlim([lower_limit, 1])
            ax.set_title(f'[{"TEST" if is_test else "VALIDATION"}] Prediction Probability Distribution at Epoch {epoch}')
            ax.set_xlabel('Prediction Probability')
            ax.set_ylabel('Number of Predictions')
            plt.tight_layout()

            return wandb.Image(plt)
        except Exception as e:
            logging.info(f'Could not generate prediction probability plot: {str(e)}')
            return None

    def _setup_data_loaders(self):
        train_data_loader, test_data_loader, val_data_loader = self.dataset.get_data_loaders(batch_size=self.args.batch_size, 
                                                                                             tokenizer=self.tokenizer)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.val_data_loader = val_data_loader
    
    def _setup_optimizer(self):
        num_train_steps = len(self.train_data_loader) * self.args.num_epochs
        optimizer, scheduler = build_optimizer(self.network,
                                               num_train_steps,
                                               self.args.learning_rate,
                                               self.args.adam_eps,
                                               self.args.warmup_steps,
                                               self.args.weight_decay)
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def _get_pretrained_network(self, model_name, model_name_or_path):
        # bert-base-uncased
        config = BertConfig.from_pretrained(model_name_or_path)
        network = BertForSequenceClassification.from_pretrained(model_name_or_path, config=config)


        if self.args.use_softmax_layer:
            new_clf = nn.Sequential(
                network.classifier,
                nn.LogSoftmax(dim=1),
            )
            network.classifier = new_clf

        return network