import pandas as pd
import torch
import torch.bin
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.helper.path_helper import *


class ModelDataset():
    def __init__(self, dataset_name, seed, max_seq_length, do_lower_case, train_frac, use_val):
        self.dataset_name = dataset_name
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.train_frac = train_frac
        self.use_val = use_val

        self.original_df = self._load_dataset(self.dataset_name)
        self.label_weights = self._calculate_label_weights()

    def get_data_loaders(self, batch_size, tokenizer):
        train_df, test_df, val_df = self._get_train_test_val()

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


    def _calculate_label_weights(self):
        total_samples = len(self.original_df['label'])
        class_counts = torch.bincount(torch.tensor(self.original_df['label']))
        label_weights = total_samples / (2*class_counts)
        label_weights = label_weights.float()
        return label_weights

    def _load_dataset(self, dataset_name):
        return pd.read_csv(f'{dataset_raw_file_path(dataset_name)}.csv')
    

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

            if file_exists_or_create(train_file_path) and file_exists_or_create(train_file_path) and check_val:
                self.train_df = pd.read_csv(train_file_path)
                self.test_df = pd.read_csv(test_file_path)
                self.validation_df = pd.read_csv(validation_file_path) if self.use_val else pd.DataFrame()
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