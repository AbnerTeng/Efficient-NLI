import os
import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from .preproc import DataProcessor
from .feature_engineering import FeatureEngineering
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    fe = FeatureEngineering(
        f"{os.getcwd()}/data_mining_final/train.csv",
        f"{os.getcwd()}/data_mining_final/test.csv"
    )
    train_data, test_data = fe.generate_cols()
    tokenizer = AutoTokenizer.from_pretrained(
        "jb2k/bert-base-multilingual-cased-language-detection",
        do_lower_case=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "jb2k/bert-base-multilingual-cased-language-detection",
        num_labels=3
    )
    ## TODO Multi-lingual preprocessor
