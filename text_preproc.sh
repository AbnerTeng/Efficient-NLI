#!/bin/bash

echo "
Generate the preprocessed text files for the training and test sets.
"

read -p "Enter the type of text file (train/test): " type

python -m src.text_preproc --type $type