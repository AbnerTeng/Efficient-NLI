"""
Data Visualization & Preprocessing
"""
import string
from typing import Union, Dict
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import sweetviz

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class TextPreproc:
    """
    Text preprocessing

    1. removing stopwords and punctuations
    2. stemming / lemmatizing
    """
    def __init__(self, data: Union[pd.DataFrame, Dict]) -> None:
        self.data = data
        self.punctuations = string.punctuation
        self.lemmatizer = WordNetLemmatizer()

    def preproc(self, text: str, lang: str) -> str:
        """
        process text
        """
        pass

class ContraData(Dataset):
    """
    Create torch.Dataset for training
    """
    def __init__(self, data_list, tokenizer) -> None:
        self.data = data_list
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        premise = self.data[idx][0]
        hypothesis = self.data[idx][1]
        label = self.data[idx][2]
        encoder = self.tokenizer(
            premise, hypothesis,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoder['input_ids'].squeeze()
        attention_mask = encoder['attention_mask'].squeeze()
        label = torch.Tensor(label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }

    def __len__(self) -> int:
        return len(self.data)


def auto_eda(data: Union[pd.DataFrame, Dict]) -> None:
    """
    (Optional)
    Using sweetviz for automation exploratory data analysis (EDA)
    sweetviz: https://pypi.org/project/sweetviz/
    
    data: (pd.DataFrame / Dict) data to be analyzed
    """
    report = sweetviz.analyze(data)
    report.show_html(
        filepath="eda_report.html",
        open_browser=False,
        layout="widescreen",
        scale=0.8
    )
