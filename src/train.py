"""
train session
"""
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class BertNLI(nn.Module):
    """
    BERT for Natural Language Inference
    """
    def __init__(self, bert_model, num_classes=3):
        super(BertNLI, self).__init__()
        self.bert_model = bert_model
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )
        self.loss = nn.CrossEntropyLoss()


    def forward(self, input_ids, attention_mask, labels=None):
        """
        forward propagation
        """
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        outputs = self.dropout(outputs.pooler_output)
        logits = self.classifier(outputs)
        return logits


class Training:
    """
    Training session
    """
    def __init__(self, train_data_loader, test_data_loader, config: dict):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.cfg = config


    def train(self):
        """
        train
        """
        self.cfg['model'].train()
        for epoch in tqdm(range(self.cfg['epoch'])):
            print(f"epoch: {epoch}")
            for batch in self.train_data_loader:
                inputs = {
                    k: v.to(self.cfg['device']) for k, v in batch.items()
                }
                labels = inputs.pop('label').to(self.cfg['device'])
                outputs = self.cfg['model'](**inputs, labels=labels)
                loss = self.cfg['criteria'](outputs, labels)
                self.cfg['optim'].zero_grad()
                loss.backward()
                self.cfg['optim'].step()
        return loss.item()


    def evaluate(self):
        """
        model evaluation
        """
        self.cfg['model'].eval()
        with torch.no_grad():
            for batch in self.test_data_loader:
                inputs = {
                    k: v.to(self.cfg['device']) for k, v in batch.items()
                }
                outputs = self.cfg['model'](**inputs)
                probs = F.softmax(outputs, dim=-1)
                predicted_labels = torch.argmax(probs) # TODO: check this
                for text, prob in zip(
                    inputs['input_ids'], probs
                ):
                    print(f"text: {text}")
                    print(f"prob: {prob}")
                    print(f"predicted label: {predicted_labels}"
                )
                