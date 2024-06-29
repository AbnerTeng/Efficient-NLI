"""
General utility functions for the project.
"""
from typing import Union, Dict
import json
import yaml
import pandas as pd
import gdown
from .constants import DATA_URL

def download_data() -> None:
    """
    Download data from the google drive using gdown
    """
    gdown.download_folder(DATA_URL, quiet=False)

def load_data(path: str) -> Union[pd.DataFrame, Dict]:
    """
    Load data from different formats
    
    path: (str) path to the data file
    
    """
    suffix = path.split('.')[-1]

    if suffix == "csv":
        data = pd.read_csv(path, encoding="utf-8")
    elif suffix == "yaml":
        with open(path, "r", encoding="utf-8") as yml_file:
            data = yaml.safe_load(yml_file)
    elif suffix == "json":
        with open(path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

    return data
