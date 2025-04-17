#%% --------------------------------------------------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_similarity(sentence1: str, sentence2: str) -> float:
    model_name = "textattack/bert-base-cased-STS-B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    similarity_score = outputs.logits.item()
    return similarity_score
#%% --------------------------------------------------------------------------------------------------------------------
s1 = "A man is playing a guitar."
s2 = "Someone is strumming a musical instrument."
score = predict_similarity(s1, s2)
print(f"Sentence 1: {s1}")
print(f"Sentence 2: {s2}")
print(f"Predicted similarity score: {score:.4f}")