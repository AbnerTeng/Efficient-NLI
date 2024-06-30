"""
Main execution script
"""
import os
import warnings
import argparse
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import transformers
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from .preproc import DataProcessor, ContraData, ContraTestData
from .train import BertNLI, Training, Classifier
from .preproc_v2 import auto_eda

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def parse_args() -> argparse.ArgumentParser:
    """
    define arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_eda",
        action="store_true"
    )
    parser.add_argument(
        "--do_preproc",
        action="store_true"
    )
    parser.add_argument(
        "--mode", type=str, default="bert"
    )
    return parser.parse_args()


def process_data(data: pd.DataFrame, classifier: Classifier) -> pd.DataFrame:
    """
    Return combined dataframe
    """
    embedded_premise, embedded_hypo = \
        classifier.embedding(data['premise']), \
        classifier.embedding(data['hypothesis'])
    premise_df, hypo_df = \
        pd.DataFrame(embedded_premise.cpu()), \
        pd.DataFrame(embedded_hypo.cpu())
    new_columns = [i for i in range(1024, 2049)]
    hypo_df.rename(
        columns=dict(
            zip(
                hypo_df.columns, new_columns
            )
        ),
        inplace=True
    )
    combined_df = pd.concat([premise_df, hypo_df], axis=1)
    return combined_df


if __name__ == "__main__":
    args = parse_args()
    _device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    train_data = f"{os.getcwd()}/data/train_clean_v2.csv"
    test_data = f"{os.getcwd()}/data/test_clean_v2.csv"
    if args.do_eda:
        auto_eda(train_data)
        auto_eda(test_data)
    if args.mode == "bert":
        bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )
        bert_model = BertModel.from_pretrained(
            'bert-base-uncased',
            num_labels=3
        )
        cfg = {
            'model': bert_model,
            'tokenizer': bert_tokenizer,
            'lr': 2e-5,
            'epoch': 3,
            'batch_size': 32,
            'gradient_accumulation_steps': 1,
            'device': _device,
            'optim': AdamW(bert_model.parameters(), lr=5e-5),
            'criteria': nn.CrossEntropyLoss(),
        }
        train_dat_proc = DataProcessor(
            train_data,
            "train",
        )
        train_list = train_dat_proc.getlist()
        contradict_dataset = ContraData(
            train_list,
            tokenizer=bert_tokenizer
        )
        train_loader = DataLoader(
            contradict_dataset, batch_size=32, shuffle=True
        )
        test_dat_proc = DataProcessor(
            test_data,
            "test"
        )
        test_list = test_dat_proc.getlist()
        contradict_datatset_test = ContraTestData(
            test_list,
            tokenizer=bert_tokenizer
        )
        test_loader = DataLoader(
            contradict_datatset_test, batch_size=32, shuffle=False
        )
        bert_nli = BertNLI(cfg['model'])
        cfg['model'] = bert_nli.to(cfg['device'])
        training = Training(train_loader, test_loader, cfg)
        training.train()
        training.evaluate()

    elif args.mode == "xlm_xgb":
        xlm_roberta_tokenizer = AutoTokenizer.from_pretrained(
            "intfloat/multilingual-e5-large"
        )
        xlm_roberta_model = AutoModel.from_pretrained(
            "intfloat/multilingual-e5-large",
            num_labels=3
        )
        cfg = {
            'model': xlm_roberta_model,
            'tokenizer': xlm_roberta_tokenizer,
            'batch_size': 32,
            'device': _device,
        }
        xgbclassifier = Classifier(cfg)
        combined_train_df = process_data(train_data, xgbclassifier)
        combined_test_df = process_data(test_data, xgbclassifier)
        prediction = xgbclassifier.classify(
            combined_train_df, combined_test_df
        )
        prediction = pd.DataFrame(prediction)
        prediction.to_csv(
            f"{os.getcwd()}/data/prediction.csv",
            index=False
        )
