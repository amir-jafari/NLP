#%% --------------------------------------------------------------------------------------------------------------------
# pip install numpy==1.26.4
#%% --------------------------------------------------------------------------------------------------------------------
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import numpy as np
import torch

def preprocess(example):
    tokenized_example = tokenizer(example['question'], example['context'], truncation=True, padding='max_length', max_length=256)
    answer = example['answers']
    if len(answer['answer_start']) > 0:
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_example['start_positions'] = tokenized_example.char_to_token(start_char) or 0
        tokenized_example['end_positions'] = tokenized_example.char_to_token(end_char - 1) or 0
    else:
        tokenized_example['start_positions'] = 0
        tokenized_example['end_positions'] = 0
    return tokenized_example

#%% --------------------------------------------------------------------------------------------------------------------
dataset = load_dataset("squad_v2")
dataset_small = dataset['train'].select(range(350))
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased')
encoded_dataset = dataset_small.map(preprocess)

#%% --------------------------------------------------------------------------------------------------------------------
training_args = TrainingArguments(output_dir='./results', num_train_epochs=1, per_device_train_batch_size=32, logging_steps=50)
trainer = Trainer(model=model, args=training_args, train_dataset=encoded_dataset, tokenizer=tokenizer)
trainer.train()

#%% --------------------------------------------------------------------------------------------------------------------
predictions = trainer.predict(encoded_dataset)
start_preds = np.argmax(predictions.predictions[0], axis=-1)
end_preds = np.argmax(predictions.predictions[1], axis=-1)

correct = 0
for i, example in enumerate(encoded_dataset):
    if example['start_positions'] == start_preds[i] and example['end_positions'] == end_preds[i]:
        correct += 1

torch.save(model.state_dict(), "qa_model.pt")
accuracy = correct / len(encoded_dataset)
print(f"Demo span accuracy on sample: {accuracy:.2%}")
