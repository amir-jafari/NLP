#%% --------------------------------------------------------------------------------------------------------------------
from datasets import load_dataset
from transformers import pipeline

#%% --------------------------------------------------------------------------------------------------------------------
imdb_test = load_dataset("imdb", split="test[:3]")
classifier = pipeline("sentiment-analysis")

#%% --------------------------------------------------------------------------------------------------------------------
for sample in imdb_test:
    text = sample["text"]
    prediction = classifier(text)[0]
    print(f"\nText:\n{text[:200]}...")
    print("Sentiment Prediction:", prediction)
