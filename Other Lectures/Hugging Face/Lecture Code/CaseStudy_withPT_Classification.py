#%% --------------------------------------------------------------------------------------------------------------------
# pip install numpy==1.26.4
#%% --------------------------------------------------------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np

def encode_labels(example):
    example['label'] = 0 if len(example['answers']['text']) == 0 else 1
    return example
def tokenize(example):
    return tokenizer(example['question'], example['context'], truncation=True, padding='max_length', max_length=128)

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.load_state_dict(torch.load("c_model.pt"))
model.eval()

#%% --------------------------------------------------------------------------------------------------------------------
dataset = load_dataset("squad_v2")
dataset_small = dataset['train'].select(range(350))
dataset_small = dataset_small.map(encode_labels)
encoded_dataset = dataset_small.map(tokenize, batched=True)

#%% --------------------------------------------------------------------------------------------------------------------
trainer = Trainer(model=model, tokenizer=tokenizer)
predictions = trainer.predict(encoded_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
accuracy = (preds == encoded_dataset['label']).mean()
print(f"Demo accuracy on sample: {accuracy:.2%}")
