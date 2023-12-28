#!/bin/bash

echo "Natural Language Inference project"


read -p "
Choose a model: 

1. bert-base-uncased with training data fine-tuning
2. XLM-RoBERTa + XGBoostClassifier

type 1 or 2: " model

if [ $model == 1]; then
    echo "Running bert-base-uncased with training data fine-tuning"
    python -m src.main --mode bert
elif [ $model == 2]; then
    echo "Running XLM-RoBERTa + XGBoostClassifier"
    python -m src.main --mode xlm_xgb
else
    echo "Invalid input"
fi
