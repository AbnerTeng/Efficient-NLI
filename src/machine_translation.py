"""
script for machine translation
"""
import warnings
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
warnings.filterwarnings("ignore")


class MachineTranslation:
    """
    Text2Text generation using Helsinki-NLP
    """
    def __init__(self, data_path: str, model_path: str) -> None:
        self.data = pd.read_csv(data_path)
        self.pretrain = json.load(open(model_path, "r", encoding="utf-8"))
        self.sup_lang = ['fr', 'ar', 'th', 'zh']
        self.tokenizers, self.models = {}, {}
        for lang in self.sup_lang:
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(self.pretrain[f"{lang}2en"])
            self.models[lang] = AutoModelForSeq2SeqLM.from_pretrained(self.pretrain[f"{lang}2en"])


    def translate(self, source_text: str, lang: str) -> str:
        """
        Translate source text to target language
        """
        tokenizer, models = self.tokenizers[lang], self.models[lang]
        inputs = tokenizer.encode(source_text, return_tensors='pt')
        greedy_outputs = models.generate(inputs, max_length=128)
        output = tokenizer.decode(greedy_outputs[0], skip_special_tokens=True)
        return output


    def main(self) -> pd.DataFrame:
        """
        Translate all source text to target language
        """
        target_premise, target_hypo = [], []
        for i in tqdm(range(len(self.data))):
            lang_abv = self.data.lang_abv.iloc[i]
            if lang_abv in self.sup_lang:
                target_premise.append(
                    self.translate(
                        self.data.premise.iloc[i],
                        lang_abv
                    )
                )
                target_hypo.append(
                    self.translate(
                        self.data.hypothesis.iloc[i],
                        lang_abv
                    )
                )
                self.data.lang_abv.iloc[i] = 'en'
                self.data.language.iloc[i] = 'English'
            else:
                target_premise.append(self.data.premise.iloc[i])
                target_hypo.append(self.data.hypothesis.iloc[i])
        return target_premise, target_hypo


if __name__ == "__main__":
    mt = MachineTranslation(
        data_path="data/train.csv",
        model_path="multilang_model.json"
    )
    pr, hyp = mt.main()
    mt.data['translated_premise'] = pr
    mt.data['translated_hypothesis'] = hyp
    mt.data.to_csv("data/train_clean_v2.csv", index=False)
