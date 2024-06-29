# Efficient-NLI

Efficient approaches for natural language inference (NLI) tasks

Authors

- [Yu-Chen (Abner) Den](https://www.abnerteng.github.io)
- [Meng-Chen (Angela) You](https://www.linkedin.com/in/meng-chun-you/)

## Set the Environment

### virtualenv

We recommend to use a virtual environment to install the dependencies.

```bash
python3 -m venv your_venv_name
source your_venv_name/bin/activate
```

```bash
pip install -r requirements.txt
```

## Dataset in-use

[Stanford Natural Language Inference (SNLI) dataset](https://nlp.stanford.edu/projects/snli/)
[Dataset card](https://huggingface.co/datasets/stanfordnlp/snli)

### Download the dataset

```bash
chmod +x scripts/download_data.sh
bash scripts/download_data.sh
```

### Preprocessing

```bash
chmod +x text_preproc.sh
./text_preproc.sh
```

## Training session

```bash
chmod +x run_classification.sh
./run_classification.sh
```

## Evaluation

### Model performance table

To be continued... I'm lazy to write this part.
