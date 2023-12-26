# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')

# def mean_pooling(model_output, attention_mask):
#     """
#     Mean Pooling - Take attention mask into account for correct averaging
#     """
#     token_embeddings = model_output[0]
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# %%
"""
BERT to classify language -> different Lang BERT
"""
# %%
import pandas as pd
# %%
pred = pd.read_csv('../data/predicted_labels.csv')
submit = pd.read_csv('../data/sample_submission.csv', index_col=0)
submit['prediction'] = pred
opt = pd.DataFrame(
    {
        'id': submit.index,
        'prediction': pred['0']
    }
)
opt.to_csv('../data/submission.csv', index=False)
# %%
data = pd.read_csv('../data/test_clean_v2.csv')

for i in range(len(data)):
    if data.lang_abv.iloc[i] in ['zh', 'ar', 'fr', 'th']:
        data.lang_abv.iloc[i] = 'en'
        data.language.iloc[i] = 'English'

data.to_csv('../data/test_clean_v2.csv', index=False)
# %%
import pandas as pd
def clean_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(
        columns=['premise', 'hypothesis']
    )
    data.rename(
        columns={
            "translated_premise": "premise",
            "translated_hypothesis": "hypothesis"
        },
        inplace=True
    )
    return data

train = pd.read_csv('../data/train_clean_v2.csv')
train = clean_columns(train)
test = pd.read_csv('../data/test_clean_v2.csv')
test = clean_columns(test)
train.to_csv('../data/train_clean_v2.csv', index=False)
test.to_csv('../data/test_clean_v2.csv', index=False)
# %%
