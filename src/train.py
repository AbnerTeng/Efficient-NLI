"""
train session
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Classifier:
    """
    XLM RoBERTa Embedding with XGBoost Classifier
    """
    def __init__(self, cfg: dict) -> None:
        self.all_embeddings = []
        self.cfg = cfg


    def embedding(self, input_text: list) -> torch.Tensor:
        """
        Get word embedding
        """
        total_batches = len(input_text) // self.cfg['batch_size']
        if len(input_text) % self.cfg['batch_size'] != 0:
            total_batches += 1
        for i in tqdm(range(total_batches)):
            start_index = i * self.cfg['batch_size']
            end_index = min((i + 1) * self.cfg['batch_size'], len(input_text))
            batch_text = input_text[start_index:end_index]
            batch_dict = self.cfg['tokenizer'](
                batch_text,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.cfg['device'])
            with torch.no_grad():
                outputs = self.cfg['model'](**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]
            self.all_embeddings.append(embeddings)

            del batch_dict
            del outputs

        embeddings_tensor = torch.cat(
            self.all_embeddings, dim=0
        )
        return embeddings_tensor


    def classify(self, data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        XGBoost Classifier
        """
        x_train, x_val, y_train, y_val = train_test_split(
            data, data['label'], test_size=0.2, random_state=42
        )
        xgbc = XGBClassifier(
            objective='multi:softprob',
            booster='gbtree',
            n_estimator=500,
            eval_metric='mlogloss',
            max_depth=6,
            random_state=42
        )
        xgbc.fit(x_train, y_train)
        y_val_pred = xgbc.predict(x_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Accuracy: {accuracy}")
        y_test_pred = xgbc.predict(test_data)
        return y_test_pred


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
                for _, prob in zip(
                    inputs['input_ids'], probs
                ):
                    predicted_labels.append(np.argmax(prob))
        predicted_labels = pd.DataFrame(predicted_labels)
        predicted_labels.to_csv(
            'data/predicted_labels.csv', index=False
        )
                