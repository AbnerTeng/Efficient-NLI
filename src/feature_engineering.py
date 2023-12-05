import os
from .utils import load_data


class FeatureEngineering:
    """
    Get new columns of language label
    """
    def __init__(self, tr_path: str, ts_path: str) -> None:
        self.train_data = load_data(tr_path)
        self.test_data = load_data(ts_path)
        self.unique_lang = self.train_data['language'].unique().tolist()


    def generate_cols(self) -> tuple:
        """
        Generate new columns
        """
        lang_map = {
            lang: idx for idx, lang in enumerate(self.unique_lang)
        }
        self.train_data['lang_label'] = self.train_data['language'].map(lang_map)
        self.test_data['lang_label'] = self.test_data['language'].map(lang_map)
        return self.train_data, self.test_data

if __name__ == "__main__":
    train_path = f"{os.getcwd()}/data_mining_final/train.csv"
    test_path = f"{os.getcwd()}/data_mining_final/test.csv"
    fe = FeatureEngineering(train_path, test_path)
    train_data, test_data = fe.generate_cols()
        