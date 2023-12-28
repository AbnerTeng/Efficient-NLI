"""
script for machine translation
"""
import warnings
import json
import pandas as pd
from tqdm import tqdm
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
warnings.filterwarnings("ignore")


class MachineTranslation:
    """
    Text2Text generation using opus-MT from Helsinki-NLP
    """
    def __init__(self, model_path: str) -> None:
        self.pretrain = json.load(open(model_path, "r", encoding="utf-8"))
        self.sup_lang = ['fr', 'ar', 'th', 'zh']
        self.tokenizers, self.models = {}, {}
        for lang in self.sup_lang:
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(self.pretrain[f"{lang}2en"])
            self.models[lang] = AutoModelForSeq2SeqLM.from_pretrained(self.pretrain[f"{lang}2en"])
        self.translator = Translator()


    def machine_translation(self, source_text: str, lang: str) -> str:
        """
        Translate source text to target language
        """
        tokenizer, models = self.tokenizers[lang], self.models[lang]
        inputs = tokenizer.encode(source_text, return_tensors='pt')
        greedy_outputs = models.generate(inputs, max_length=128)
        output = tokenizer.decode(greedy_outputs[0], skip_special_tokens=True)
        return output


    def impute_by_googletrans(self, source_text: str) -> str:
        """
        Impute missing translation using googletrans
        """
        output = self.translator.translate(source_text, dest='en').text
        return output


    def get_output(self, data: pd.DataFrame) -> tuple:
        """
        Translate all source text to target language
        """
        target_premise, target_hypo = [], []
        for i in tqdm(range(len(data))):
            lang_abv = data.lang_abv.iloc[i]
            if lang_abv in self.sup_lang:
                target_premise.append(
                    self.machine_translation(
                        data.premise.iloc[i],
                        lang_abv
                    )
                )
                target_hypo.append(
                    self.machine_translation(
                        data.hypothesis.iloc[i],
                        lang_abv
                    )
                )
                data.lang_abv.iloc[i] = 'en'
                data.language.iloc[i] = 'English'
            else:
                target_premise.append(
                    self.impute_by_googletrans(
                        data.premise.iloc[i]
                    )
                )
                target_hypo.append(
                    self.impute_by_googletrans(
                        data.hypothesis.iloc[i]
                    )
                )
        return target_premise, target_hypo


    def main(self, data_path: str) -> None:
        """
        main function
        """
        data = pd.read_csv(data_path, encoding="utf-8")
        pr, hyp = self.get_output(data)
        data['translated_premise'] = pr
        data['translated_hypothesis'] = hyp
        data.to_csv(f"{data_path.split('.')[0]}_clean_v2.csv", index=False)


if __name__ == "__main__":
    mt = MachineTranslation(
        model_path="multilang_model.json"
    )
    mt.main("data/train.csv")
    print("Train data transaltion done!")
    mt.main("data/test.csv")
    print("Test data transaltion done!")
