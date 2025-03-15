#%% --------------------------------------------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#%% --------------------------------------------------------------------------------------------------------------------
model_name = "textattack/bert-base-cased-STS-B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#%% --------------------------------------------------------------------------------------------------------------------
sentence1 = "A man is playing a guitar."
sentence2 = "Someone is strumming a musical instrument."
inputs = tokenizer(sentence1, sentence2, return_tensors="pt")
outputs = model(**inputs)
score = outputs.logits.item()
print(score)