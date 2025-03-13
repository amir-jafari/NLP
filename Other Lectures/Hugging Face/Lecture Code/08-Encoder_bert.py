#%% --------------------------------------------------------------------------------------------------------------------
from transformers import BertTokenizer, BertForSequenceClassification
import torch

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
inputs = tokenizer('Hello world!', return_tensors='pt')
outputs = model(**inputs)
logits = outputs.logits
pred = torch.argmax(logits, dim=-1)
print(pred)