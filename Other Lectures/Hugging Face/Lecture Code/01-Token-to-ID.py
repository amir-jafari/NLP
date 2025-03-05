#%% --------------------------------------------------------------------------------------------------------------------
from transformers import BertTokenizer

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Hello world!"
ids = tokenizer.encode(text)
print(ids)
decoded = tokenizer.decode(ids)
print(decoded)
