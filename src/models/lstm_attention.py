import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm

import wandb
from src.data.dataset import ModelDataset
from src.helper.logging_helper import setup_logging
from src.helper.path_helper import *
from src.helper.seed_helper import initialize_gpu_seed

setup_logging()



class LSTMWithAttention(nn.Module):
    # def __init__(self, input_size, hidden_size, output_size, seed, num_layers=1, bidirectional=True, checkpoint_after_n=100):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(LSTMWithAttention, self).__init__()
        self.cur_date_prefix = datetime.now().strftime('%d-%m-%y_')

        self.args = args
        self.model_seed = self.args.model_seed
        self.device, _ = initialize_gpu_seed(self.model_seed)
        self.dataset = ModelDataset(dataset_name=self.args.dataset_name,
                            seed=self.args.seed,
                            max_seq_length=self.args.max_seq_length,
                            do_lower_case=True,
                            train_frac=0.8,
                            use_val=self.args.use_validation_set,
                            device=self.device,
                            is_encoded=self.args.is_encoded)

        # setup with args
 

        self.use_val = self.args.use_validation_set

        self.bn = nn.BatchNorm1d(hidden_size, device=self.device)
        self.dropout = nn.Dropout(0.7).to(self.device)
        self.num_directions = 2 if self.args.bidirectional else 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = self.args.num_layers
        self.seed = self.args.seed
        self.checkpoint_after_n = None #self.args.checkpoint_after_n
        self.bidirectional = self.args.bidirectional
        print(f'Bidirectional: {self.bidirectional}')

        if not self.args.is_encoded:
            self.embedding = nn.Embedding(len(self.dataset.vocab), self.input_size, device=self.device)
        self.lstm = nn.LSTM(self.input_size, 
                            self.hidden_size, 
                            self.num_layers, 
                            batch_first=True, 
                            bidirectional=self.bidirectional, 
                            dropout=0.7)
        
        self.lstm.to(self.device)
        self.to(device=self.device)
        self.attention = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, device=self.device)
        self.context_vector = nn.Linear(self.hidden_size * self.num_directions, 1, bias=False, device=self.device)
        
        self.fc = nn.Linear(self.hidden_size * self.num_directions, self.output_size, device=self.device)

        self.lr = self.args.learning_rate
        self.num_epochs = self.args.num_epochs

        # self.init_weights()


        self._setup_data_loaders()

        self.weight_decay = self.args.weight_decay
        self.optimizer = optim.AdamW(self.lstm.parameters(), lr=self.args.learning_rate, weight_decay=self.weight_decay)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=3)
        steps_per_epoch = len(self.train_data_loader)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=0.001, steps_per_epoch=steps_per_epoch, epochs=self.num_epochs)
        # self.loss_fn = nn.BCELoss(weight=self.dataset.label_weights)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.dataset.label_weights[1]).to(self.device)

        # self._setup_optimizer()
        self._reset_prediction_buffer()

    
    def forward(self, input_, h_n_, c_n_):
        # import code; code.interact(local=dict(globals(), **locals()))
        # input_ = input_.unsqueeze(1)
        if not self.args.is_encoded:
            input_ = self.embedding(input_)
        else:
            input_ = input_.unsqueeze(1)
        
        # import code; code.interact(local=dict(globals(), **locals()))
        output, (h_n, c_n) = self.lstm(input_, (h_n_, c_n_))

        # batch_size, seq_len, hidden_size = output.size()
        # output = output.contiguous().view(-1, hidden_size)  # Reshape to (batch_size * seq_len, hidden_size)
        # output = self.bn(output)
        # output = output.view(batch_size, seq_len, hidden_size)
        
        # attn_weights = F.softmax(self.context_vector(torch.tanh(self.attention(output))), dim=1)
        attn_weights = torch.softmax(self.attention(output), dim=1)
        context = torch.sum(attn_weights * output, dim=1)  # Context vector shape: (batch_size, hidden_dim)
        
        out = self.fc(self.dropout(context))  # Final output shape: (batch_size, output_dim)
        return out, (h_n, c_n)

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    def init_hidden(self, batch_size):
        # h_0 = Variable(torch.zeros(self.num_layers, self.args.batch_size, self.hidden_size, device=self.device))
        # c_0 = Variable(torch.zeros(self.num_layers, self.args.batch_size, self.hidden_size, device=self.device))
        h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)
        return h_0, c_0
    

    def train_model(self):
        total_loss, prev_epoch_loss = 0, 0
        best_val_acc = 0
        for epoch in tqdm(range(self.num_epochs)):
            self.train()
            sample_correct, sample_count = 0, 0
            for i, data in enumerate(self.train_data_loader):

                # todo
                ids = data['encoding'].to(self.device)
                labels = data['labels'].to(self.device)
                queries = data['query']
                query_id = data['query_id']
            
                self.optimizer.zero_grad()
                h_0, c_0 = self.init_hidden(batch_size=ids.size(0))

                output, _ = self(ids, h_0, c_0)
                # import code; code.interact(local=dict(globals(), **locals()))
                loss = self.loss_fn(output.squeeze(1), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=1.0)

                total_loss += loss.item()

                predicted = torch.sigmoid(output.squeeze(1)) >= 0.5
                sample_correct += (predicted == labels).sum().item()
                sample_count += labels.size(0)


                self.optimizer.step()
                self.scheduler.step()
                if (self.checkpoint_after_n is not None) and (i % self.checkpoint_after_n == 0):
                    print(f'Epoch: {epoch}, Loss: {loss.item()}')
                    self.save_checkpoint(epoch)
            
            train_loss = round((total_loss - prev_epoch_loss) / len(self.train_data_loader), 4)
            train_acc = round(sample_correct / sample_count, 4)
            self._wandb_log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'lr': self.scheduler.get_last_lr()[0]
            })

            prev_epoch_loss = total_loss

            # Run test+val set after each epoch
            self.test_model(epoch)
            if self.use_val:
                val_acc, val_loss = self.validate_model(epoch)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f'Epoch: {epoch}, Loss: {loss.item()}')
                    self.save_checkpoint(epoch)
                # self.scheduler.step(val_loss)
    
    def test_model(self, epoch=0):
        self._reset_prediction_buffer()
        self.lstm.eval()
        total_loss = 0
        with torch.no_grad():
            sample_correct, sample_count = 0, 0
            for i, data in tqdm(enumerate(self.test_data_loader), desc=f'[TESTING] Running epoch {epoch} ...',
                                      total=len(self.test_data_loader)):
                ids = data['encoding'].to(self.device)
                labels = data['labels'].to(self.device)
                queries = data['query']
                query_id = data['query_id']

                h_0, c_0 = self.init_hidden(batch_size=ids.size(0))
                output, _ = self(ids, h_0, c_0)
                loss = self.loss_fn(output.squeeze(1), labels)
                total_loss += loss.item()

                prediction_proba = torch.sigmoid(output)
                predictions = prediction_proba >= 0.5
                sample_correct += (predictions[0] == labels).sum().item()
                sample_count += labels.size(0)

                self.log_predictions(labels, query_id, predictions, prediction_proba, step=i, is_test=True)
            
            test_loss = round(total_loss / len(self.test_data_loader), 4)
            test_acc = round(sample_correct / sample_count, 4)
            self._wandb_log({
                'epoch': epoch,
                'test_loss': test_loss,
                'test_acc': test_acc,
            })
        logging.info(f"[Epoch {epoch}/{self.args.num_epochs}]\tTest Loss: {test_loss}\tTest Accuracy: {test_acc}")
        self.save_test_predictions(epoch)

    def validate_model(self, epoch=0):
        self._reset_prediction_buffer(is_test=False)
        self.lstm.eval()
        total_loss = 0
        with torch.no_grad():
            sample_correct, sample_count = 0, 0
            for i, data in tqdm(enumerate(self.val_data_loader), desc=f'[VALIDATE] Running epoch {epoch} ...',
                                      total=len(self.val_data_loader)):
                ids = data['encoding'].to(self.device)
                labels = data['labels'].to(self.device)
                queries = data['query']
                query_id = data['query_id']

                h_0, c_0 = self.init_hidden(batch_size=ids.size(0))
                output, _ = self(ids, h_0, c_0)
                loss = self.loss_fn(output.squeeze(1), labels)
                total_loss += loss.item()
                
                prediction_proba = torch.sigmoid(output.squeeze(1))
                predictions = prediction_proba >= 0.5
                sample_correct += (predictions[0] == labels).sum().item()
                sample_count += labels.size(0)

                self.log_predictions(labels, query_id, predictions, prediction_proba, step=i, is_test=False)
            
            val_loss = round(total_loss / len(self.val_data_loader), 4)
            val_acc = round(sample_correct / sample_count, 4)
            self._wandb_log({
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
        logging.info(f"[Epoch {epoch}/{self.args.num_epochs}]\Val Loss: {val_loss}\Val Accuracy: {val_acc}")
        # self.save_test_predictions(epoch)
        return val_acc, val_loss

    def log_predictions(self, labels, query_ids, predictions, prediction_proba, step=0, is_test=True):
        def tensor_to_list(tensor_data):
            return tensor_data.detach().cpu().numpy().reshape(-1).tolist()

        batch_size = self.test_data_loader.batch_size
        num_samples = len(self.test_data_loader.dataset) if is_test else len(self.val_data_loader.dataset)

        start_idx = step * batch_size
        end_idx = np.min([((step + 1) * batch_size), num_samples])
        
        # import code; code.interact(local=dict(globals(), **locals()))
        self.prediction_buffer['queries'][start_idx:end_idx] = query_ids
        self.prediction_buffer['labels'][start_idx:end_idx] = tensor_to_list(labels)
        self.prediction_buffer['predictions'][start_idx:end_idx] = tensor_to_list(predictions)
        self.prediction_buffer['prediction_proba'][start_idx:end_idx] = tensor_to_list(prediction_proba)

    def save_test_predictions(self, epoch):
        file_name = "".join([self.args.model_name, '__prediction_log__ep', str(epoch), '.csv'])
        log_path = experiment_file_path(self.args.experiment_name, file_name)

        file_exists_or_create(log_path)

        with open(log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.prediction_buffer.keys())
            writer.writerows(zip(*self.prediction_buffer.values()))

    def _reset_prediction_buffer(self, is_test=True):
        num_test_samples = len(self.test_data_loader.dataset) if is_test else len(self.val_data_loader.dataset)
        self.prediction_buffer = {
            'queries': np.zeros(num_test_samples, dtype=str),
            'labels': np.zeros(num_test_samples, dtype=int),
            'predictions': np.zeros(num_test_samples, dtype=int),
            'prediction_proba': np.zeros(num_test_samples, dtype=float)
        }

    def save_checkpoint(self, epoch):        
        path = f'models/{self.experiment_name}_lstm/'

        path = Path(path)
        path_exists = path.exists()

        if not path_exists:
            path.mkdir(exist_ok=True, parents=True)

        model_name = f'{self.cur_date_prefix}_lstm_epoch-{epoch}'
        # print(f'{path}/{model_name}.pt')
        torch.save(self, f'{path}/{model_name}.pt')
    
    def _wandb_log(self, wandb_dict: dict):
        if self.args.wandb:
            wandb.log(wandb_dict)

    def _setup_data_loaders(self):
        train_data_loader, test_data_loader, val_data_loader = self.dataset.get_data_loaders_lstm(batch_size=self.args.batch_size)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.val_data_loader = val_data_loader