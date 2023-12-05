"""
Main execution script
"""
import os
import warnings
import argparse
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import transformers
from transformers import BertTokenizer, BertModel
from .preproc import DataProcessor, ContraData, ContraTestData
from .train import BertNLI, Training
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def arguments() -> argparse.ArgumentParser:
    """
    define arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", type=str, default="train",
        help="train or test"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )
    _device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    cfg = {
        'model': model,
        'lr': 2e-5,
        'epoch': 3,
        'batch_size': 32,
        'gradient_accumulation_steps': 1,
        'device': _device,
        'optim': AdamW(model.parameters(), lr=5e-5),
        'criteria': nn.CrossEntropyLoss(),
    }
    # if args.type == "train":
    train_dat_proc = DataProcessor(
        f"{os.getcwd()}/data_mining_final/train.csv",
        "train"
    )
    train_list = train_dat_proc.getlist()
    contradict_dataset = ContraData(
        train_list,
        tokenizer=tokenizer
    )
    train_loader = DataLoader(
        contradict_dataset, batch_size=32, shuffle=True
    )
    # elif args.type == "test":
    test_dat_proc = DataProcessor(
        f"{os.getcwd()}/data_mining_final/test.csv",
        "test"
    )
    test_list = test_dat_proc.getlist()
    contradict_datatset_test = ContraTestData(
        test_list,
        tokenizer=tokenizer
    )
    test_loader = DataLoader(
        contradict_datatset_test, batch_size=32, shuffle=False
    )
    bert_nli = BertNLI(cfg['model'])
    cfg['model'] = bert_nli.to(cfg['device'])
    training = Training(train_loader, test_loader, cfg)
    # items = training.train()
    training.evaluate()
    
    