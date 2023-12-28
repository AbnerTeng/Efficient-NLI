"""
Data Preprocessing
"""
import torch
from torch.utils.data import Dataset
from .utils import load_data


class DataProcessor:
    """
    Processing data from .csv file
    """
    def __init__(self, dat_path: str, _type: str) -> None:
        self.data = load_data(dat_path)
        self.type = _type


    def getlist(self) -> list:
        """
        turn dataframe into list
        """
        if self.type == "train":
            data_list = self.data[
                [
                    'premise', 'hypothesis', 'label'
                ]
            ].values.tolist()
        elif self.type == "test":
            data_list = self.data[
                [
                    'premise', 'hypothesis'
                ]
            ].values.tolist()
        return data_list


class ContraData(Dataset):
    """
    create dataset for training
    """
    def __init__(self, data_list, tokenizer) -> None:
        self.data = data_list
        self.tokenizer = tokenizer


    def __getitem__(self, idx: int) -> dict:
        premise = self.data[idx][0]
        hypothesis = self.data[idx][1]
        label = self.data[idx][2]
        encoding = self.tokenizer(
            premise, hypothesis,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = torch.Tensor(label, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


    def __len__(self) -> int:
        return len(self.data)


class ContraTestData(Dataset):
    """
    create dataset for testing
    """
    def __init__(self, data_list, tokenizer) -> None:
        self.data = data_list
        self.tokenizer = tokenizer


    def __getitem__(self, idx: int) -> dict:
        premise = self.data[idx][0]
        hypothesis = self.data[idx][1]
        encoding = self.tokenizer(
            premise, hypothesis,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


    def __len__(self) -> int:
        return len(self.data)