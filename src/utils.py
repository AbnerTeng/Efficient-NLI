"""
General utility functions for the project.
"""
from typing import Union, Dict, List
import json
import yaml
import pandas as pd

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

def parse_json_file(data: List[Dict[str, Union[str, List[str]]]]) -> List[Dict[str, str]]:
    """
    Parsing JSON file

    data: (Dict) JSON data

    Initial kv pairs in use:
    - gold_label (y)
    - sentence1 (premise)
    - sentence2 (hypothesis)
    """
    new_data = list(
        map(
            lambda x: {
                k: v for k, v in eval(x).items() if k in ["gold_label", "sentence1", "sentence2"]
            },
            data
        )
    )
    return new_data
