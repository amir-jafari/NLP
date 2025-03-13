#%% --------------------------------------------------------------------------------------------------------------------
# pip install numpy==1.26.4
#%% --------------------------------------------------------------------------------------------------------------------
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch

def encode_labels(example):
    example['label'] = 0 if len(example['answers']['text']) == 0 else 1
    return example
def tokenize(example):
    return tokenizer(example['question'], example['context'], truncation=True, padding='max_length', max_length=128)

#%% --------------------------------------------------------------------------------------------------------------------
dataset = load_dataset("squad_v2")
dataset_small = dataset['train'].select(range(350))
dataset_small = dataset_small.map(encode_labels)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
encoded_dataset = dataset_small.map(tokenize, batched=True)

#%% --------------------------------------------------------------------------------------------------------------------
training_args = TrainingArguments(output_dir='./results',num_train_epochs=1,per_device_train_batch_size=32,evaluation_strategy="no",logging_steps=50,)
trainer = Trainer(model=model,args=training_args,train_dataset=encoded_dataset,tokenizer=tokenizer)
trainer.train()

#%% --------------------------------------------------------------------------------------------------------------------
predictions = trainer.predict(encoded_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
torch.save(model.state_dict(), "c_model.pt")
accuracy = (preds == encoded_dataset['label']).mean()
print(f"Demo accuracy on sample: {accuracy:.2%}")
