"""
train session
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class LangDetector(nn.Module):
    """
    Bert for Language Detection
    """
    def __init__(self, bert_model, num_classes=15) -> None:
        super(LangDetector, self).__init__()
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
        model_loss = []
        self.cfg['model'].train()
        for epoch in tqdm(range(self.cfg['epoch'])):
            print(f"epoch: {epoch}")
            for batch in tqdm(self.train_data_loader):
                inputs = {
                    k: v.to(self.cfg['device']) for k, v in batch.items()
                }
                labels = inputs.pop('label').to(self.cfg['device'])
                outputs = self.cfg['model'](**inputs, labels=labels)
                loss = self.cfg['criteria'](outputs, labels)
                self.cfg['optim'].zero_grad()
                loss.backward()
                self.cfg['optim'].step()
                model_loss.append(loss.item())
            train_loss = sum(model_loss) / len(model_loss)
            print(f"[ Train | {epoch + 1:03d} / {self.cfg['epoch']:03d} ]|loss = {train_loss:.5f}")


    def evaluate(self):
        """
        model evaluation
        """
        self.cfg['model'].eval()
        predicted_labels = []
        with torch.no_grad():
            for batch in tqdm(self.test_data_loader):
                inputs = {
                    k: v.to(self.cfg['device']) for k, v in batch.items()
                }
                outputs = self.cfg['model'](**inputs)
                probs = F.softmax(outputs, dim=-1).cpu().numpy()
                for text, prob in zip(
                    inputs['input_ids'], probs
                ):
                    # print(f"text: {text}")
                    # print(f"prob: {prob}")
                    # print(f"predicted label: {np.argmax(prob)}")
                    predicted_labels.append(np.argmax(prob))
        predicted_labels = pd.DataFrame(predicted_labels)
        predicted_labels.to_csv(
            'data/predicted_labels.csv', index=False
        )
                