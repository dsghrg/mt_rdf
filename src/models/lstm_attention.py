from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import wandb
from src.data.dataset import ModelDataset
from src.helper.seed_helper import initialize_gpu_seed


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
                            device=self.device)

        # setup with args



        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = self.args.num_layers
        self.seed = self.args.seed
        self.checkpoint_after_n = self.args.checkpoint_after_n
        self.bidirectional = self.args.bidirectional

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_vector = nn.Linear(self.hidden_size, 1, bias=False)
        
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)

        self.lr = self.args.learning_rate
        self.num_epochs = self.args.num_epochs



        self._setup_data_loaders()

        self.optimizer = optim.Adam(self.lstm.parameters(), lr=self.args.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.dataset.label_weights)

        # self._setup_optimizer()
        # self._reset_prediction_buffer()
    
    def forward(self, input_, h_n_, c_n_):
        output, (h_n, c_n) = self.lstm(input_, (h_n_, c_n_))

        # h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # lstm_out, _ = self.lstm(x, (h_0, c_0))  # LSTM output shape: (batch_size, seq_len, hidden_dim)
        
        attn_weights = F.softmax(self.context_vector(torch.tanh(self.attention(output))), dim=1)
        context = torch.sum(attn_weights * output, dim=1)  # Context vector shape: (batch_size, hidden_dim)
        
        out = self.fc(context)  # Final output shape: (batch_size, output_dim)
        return out, (h_n, c_n)

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, self.hidden_size, device=self.device))
        c_0 = Variable(torch.zeros(1, self.hidden_size, device=self.device))
        return h_0, c_0
    

    def train(self):
        self.lstm.train()
        total_loss, prev_epoch_loss = 0
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.train_data_loader):

                # todo
                # ids = data['input_ids'].to(self.device)
                labels = data['labels'].to(self.device)
                queries = data['query']
                query_id = data['query_id']
            
                self.optimizer.zero_grad()
                h_0, c_0 = self.init_hidden()
                output, _ = self.forward(queries, h_0, c_0)
                loss = self.loss_fn(output, labels)
                loss.backward()

                total_loss += loss.item()


                self.optimizer.step()
                if i % self.checkpoint_after_n == 0:
                    print(f'Epoch: {epoch}, Loss: {loss.item()}')
                    self.save_checkpoint(epoch)
            
            train_loss = round((total_loss - prev_epoch_loss) / len(self.train_data_loader), 4)
            # train_acc = round(sample_correct / sample_count, 4)
            self._wandb_log({
                'epoch': epoch,
                'train_loss': train_loss,
                # 'train_acc': train_acc
            })

            prev_epoch_loss = total_loss


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
        train_data_loader, test_data_loader, val_data_loader = self.dataset.get_data_loaders_lstm(batch_size=self.args.batch_size, 
                                                                                             tokenizer=self.tokenizer)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.val_data_loader = val_data_loader