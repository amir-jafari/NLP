#%% --------------------------------------------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def classify_text(text: str):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(dim=1).item()
    label_names = ["negative", "positive"]
    return label_names[predicted_class_id]

#%% --------------------------------------------------------------------------------------------------------------------
sample_text = "I love this movie!"
prediction = classify_text(sample_text)
print(f"Input Text: {sample_text}")
print(f"Predicted Sentiment: {prediction}")