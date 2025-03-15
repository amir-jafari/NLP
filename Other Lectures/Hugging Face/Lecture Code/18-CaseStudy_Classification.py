#%% --------------------------------------------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#%% --------------------------------------------------------------------------------------------------------------------
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#%% --------------------------------------------------------------------------------------------------------------------
text = "I love this movie!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_label = logits.argmax(dim=1).item()
print(f"The predicted_label for the text 'I love this movie!' is {predicted_label}.")