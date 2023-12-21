"""
remove stopwords and punctuations
stemming / lemmatizing
"""
import string
import argparse
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


class TextPreproc:
    """
    Text preprocessing
    
    1. removing stopwords and punctuations
    2. stemming / lemmatizing
    """
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.punctuations = string.punctuation
        self.lemmatizer = WordNetLemmatizer()
        self.processed_premise, self.processed_hypo = [], []


    def preproc(self, text: str, lang: str) -> str:
        """
        process text
        """
        stop_words = set(stopwords.words(lang))
        words = nltk.word_tokenize(text)
        filtered_words = [
            self.lemmatizer.lemmatize(word) \
            for word in words if (word not in stop_words) and (word not in self.punctuations)
        ]
        processed_words = " ".join(filtered_words)
        return processed_words


    def getlist(self) -> tuple:
        """
        get the processed list
        """
        for idx in tqdm(range(len(self.data))):
            text_lang = self.data.language.iloc[idx].lower()
            if text_lang in stopwords.fileids():
                self.processed_premise.append(
                    self.preproc(
                        self.data.premise.iloc[idx], text_lang
                    )
                )
                self.processed_hypo.append(
                    self.preproc(
                        self.data.hypothesis.iloc[idx], text_lang
                    )
                )
            else:
                self.processed_premise.append(
                    self.data.premise.iloc[idx]
                )
                self.processed_hypo.append(
                    self.data.hypothesis.iloc[idx]
                )
        return self.processed_premise, self.processed_hypo


def parse_args():
    """
    parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", type=str, default="train"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    clean_text = []
    if args.type == "train":
        df = pd.read_csv(f'data/{args.type}.csv')
        processor = TextPreproc(df)
    elif args.type == "test":
        df = pd.read_csv(f'data/{args.type}.csv')
        processor = TextPreproc(df)
    premise, hypo = processor.getlist()
    df['premise'] = premise
    df['hypothesis'] = hypo
    df.to_csv(f'data/{args.type}_clean.csv', index=False)
