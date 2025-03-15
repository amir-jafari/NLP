#%% --------------------------------------------------------------------------------------------------------------------
import torch
from transformers import BertTokenizer, BertForSequenceClassification

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.eval()

#%% --------------------------------------------------------------------------------------------------------------------
text = "Hello NLP class - Hugging Face Lecture!"
inputs = tokenizer(text, return_tensors="pt")

#%% --------------------------------------------------------------------------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()
print(f"Input text: {text}")
print(f"Predicted label: {predicted_label}")
