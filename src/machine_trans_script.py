# %%
import pandas as pd
test = pd.read_csv('../data/test.csv')
text = test[test['lang_abv'] == "th"].premise.iloc[0]
# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-th-en')
model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-th-en')

def machine_translation(source_text: str) -> str:
    inputs = tokenizer.encode(source_text, return_tensors='pt')
    greedy_outputs = model.generate(inputs, max_length=128)
    return tokenizer.decode(greedy_outputs[0], skip_special_tokens=True)

print(machine_translation(text))
# %%
import json

dic = json.load(open('../multilang_model.json', 'r'))
# %%
