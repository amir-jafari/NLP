#%% --------------------------------------------------------------------------------------------------------------------
import torch
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel

#%% --------------------------------------------------------------------------------------------------------------------
class CustomClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_emb = outputs.last_hidden_state[:,0,:]
        logits = self.classifier(cls_token_emb)
        return logits

#%% --------------------------------------------------------------------------------------------------------------------
model = CustomClassifier()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer("I love using Hugging Face!", return_tensors="pt")
with torch.no_grad():
    logits = model(input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"])
    pred = torch.argmax(logits, dim=-1)
    print(pred)