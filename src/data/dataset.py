import ast
import logging
import re

import pandas as pd
import torch
import torch.bin
from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import DataLoader, Dataset

from src.helper.logging_helper import *
from src.helper.path_helper import *
from src.models.config import *

setup_logging()


class ModelDataset():
    def __init__(self, dataset_name, seed, max_seq_length, do_lower_case, train_frac, use_val, device, is_encoded):
        self.dataset_name = dataset_name
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.train_frac = train_frac
        self.use_val = use_val
        self.device = device
        self.is_encoded = is_encoded

        self.encoding = self._load_encoding()
        if not self.is_encoded:
            self.vocab = {"<PAD>": 0, "SELECT": 1, "WHERE": 2, "ASK": 3, "FILTER": 4, "?s": 5, "?p": 6, "?o": 7, "{": 8, "}": 9,
                            ".": 10, "AND": 11}
        
        self.original_df = self._load_dataset()
        self.label_weights = self._calculate_label_weights()


    def get_data_loaders(self, batch_size, tokenizer):
        train_df, test_df, val_df = self._get_train_test_val()
        train_df.to_csv(dataset_processed_file_path(self.dataset_name, f'train.csv', seed=self.seed), index=False)
        test_df.to_csv(dataset_processed_file_path(self.dataset_name, f'test.csv', seed=self.seed), index=False)
        val_df.to_csv(dataset_processed_file_path(self.dataset_name, f'val.csv', seed=self.seed), index=False)

        train_ds = PytorchDataset(data_df=train_df, max_seq_length=self.max_seq_length, tokenizer=tokenizer)
        test_ds = PytorchDataset(data_df=test_df, max_seq_length=self.max_seq_length, tokenizer=tokenizer)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        if val_df.empty:
            val_dl = None
        else:
            val_ds = PytorchDataset(data_df=val_df, max_seq_length=self.max_seq_length, tokenizer=tokenizer)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)


        return train_dl, test_dl, val_dl
    
    def get_data_loaders_lstm(self, batch_size):
        train_df, test_df, val_df = self._get_train_test_val()
        train_df.to_csv(dataset_processed_file_path(self.dataset_name, f'train.csv', seed=self.seed), index=False)
        test_df.to_csv(dataset_processed_file_path(self.dataset_name, f'test.csv', seed=self.seed), index=False)
        val_df.to_csv(dataset_processed_file_path(self.dataset_name, f'val.csv', seed=self.seed), index=False)
        
        train_ds = LSTMDataset(data_df=train_df)
        test_ds = LSTMDataset(data_df=test_df)


        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        if val_df.empty:
            val_dl = None
        else:
            val_ds = LSTMDataset(data_df=val_df)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)


        return train_dl, test_dl, val_dl


    def _calculate_label_weights(self):
        total_samples = len(self.original_df['label'])
        class_counts = torch.bincount(torch.tensor(self.original_df['label']).to(self.device))
        label_weights = total_samples / (2*class_counts)
        label_weights = label_weights.float()
        return label_weights

    def _load_dataset(self):
        logging.info(f'Loading dataset {self.dataset_name}')

        raw_file = pd.read_csv(dataset_raw_file_path(Config.DATASET[self.dataset_name]))
        if not self.is_encoded:
            self.vocab = self.build_vocab(raw_file['query'], self.vocab)
            self.encoding = [self.tokenize_and_convert_to_ids(query) for query in raw_file['query']]
            max_len = max(len(tokens) for tokens in self.encoding)
            self.encoding = [torch.tensor(tokens + [self.vocab["<PAD>"]] * (max_len - len(tokens)), device=self.device) for tokens in self.encoding]
        # import code; code.interact(local=dict(globals(), **locals()))


        raw_file['encoding'] = pd.Series(self.encoding)

        raw_file = raw_file[raw_file['label'] >= 0]
        raw_file.reset_index(drop=True, inplace=True)
        return raw_file
    
    def _load_encoding(self):
        if self.is_encoded:
            return torch.load(dataset_processed_file_path(self.dataset_name, f'{self.dataset_name}_encoding.pt', self.seed))
        return None
    

    def _get_train_test_val(self):
        try:
            return self.train_df, self.test_df, self.val_df
        except AttributeError:
            train_file_path = dataset_processed_file_path(self.dataset_name, f'train.csv',
                                                          seed=self.seed)
            test_file_path = dataset_processed_file_path(self.dataset_name, f'test.csv',
                                                         seed=self.seed)
            validation_file_path = dataset_processed_file_path(self.dataset_name, f'val.csv',
                                                         seed=self.seed)

            check_val = file_exists_or_create(validation_file_path) if self.use_val else True

            if False and file_exists_or_create(train_file_path) and file_exists_or_create(train_file_path) and check_val:
                train_df = pd.read_csv(train_file_path)
                test_df = pd.read_csv(test_file_path)
                val_df = pd.read_csv(validation_file_path) if self.use_val else pd.DataFrame()
            else:

                train_df, test_df = train_test_split(self.original_df, 
                                                    train_size=self.train_frac, 
                                                    random_state=self.seed, 
                                                    stratify=self.original_df['label'])
                val_df = pd.DataFrame()
                if self.use_val:
                    test_df, val_df = train_test_split(test_df, train_size=0.5, random_state=self.seed, stratify=test_df['label'])
            return train_df, test_df, val_df


    # def _get_random_split(self, df: pd.DataFrame, train_frac: float):
    #     train_df = df.sample(frac=train_frac, random_state=self.seed)
    #     test_df = df.drop(train_df.index)
    #     val_df = pd.DataFrame()
    #     if self.use_val:
    #         # split the validation set as half of the test set, i.e.
    #         # both test and valid sets will be of the same size
    #         #
    #         val_df = test_df.sample(frac=0.5, random_state=self.seed)
    #         test_df = test_df.drop(val_df.index)
    #     return train_df, test_df, val_df

    # Define a function to tokenize while keeping <...> tokens intact
    def custom_tokenize(self, query):
        pattern = re.compile(r'<.*?>[*+?]?')
        tokens = []
        last_end = 0
        for match in pattern.finditer(query):
            start, end = match.span()
            if start > last_end:
                tokens.extend(query[last_end:start].split())
            tokens.append(query[start:end])
            last_end = end
        if last_end < len(query):
            tokens.extend(query[last_end:].split())
        return tokens

    # Add all unique tokens within angle brackets to the vocabulary
    def build_vocab(self, queries, vocab):
        unique_tokens = set()
        for query in queries:
            unique_tokens.update(self.custom_tokenize(query))
        for token in unique_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        return vocab

    # Tokenize the queries and convert tokens to indices
    def tokenize_and_convert_to_ids(self, query):
        tokens = self.custom_tokenize(query)
        token_ids = [self.vocab.get(token, self.vocab["<PAD>"]) for token in tokens]
        # import code; code.interact(local=dict(globals(), **locals()))

        return token_ids


class LSTMDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame):
        self.data_df = data_df
        # self.targets = torch.tensor(np.array(targets.values.tolist()))

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        query = row['query']
        label = row['label']
        q_id = row['query_id']
        encoding = row['encoding']

        return {
            # TODO: check how to encode input for LSTM
            'encoding': encoding,
            'labels': torch.tensor(label, dtype=torch.float),
            'query': query,
            'query_id': q_id
        }

    def __len__(self):
        return len(self.data_df)
class PytorchDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, tokenizer, max_seq_length: int):
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # self.label_list = sorted(self.idx_df.label.unique())

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        query = row['query']
        label = row['label']
        q_id = row['query_id']

        inputs = self.tokenizer.encode_plus(
            query,
            None,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'query': query,
            'query_id': q_id
        }