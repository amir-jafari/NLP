#%% --------------------------------------------------------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
#%% --------------------------------------------------------------------------------------------------------------------
model_name = "distilbert-base-uncased"
dataset = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))
tokenizer = AutoTokenizer.from_pretrained(model_name)
#%% --------------------------------------------------------------------------------------------------------------------
small_train = small_train.map(tokenize_function, batched=True).rename_column("label", "labels")
small_test = small_test.map(tokenize_function, batched=True).rename_column("label", "labels")

small_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
small_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model_fp32 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model_fp16 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

#%% --------------------------------------------------------------------------------------------------------------------
training_args_fp32 = TrainingArguments(
    output_dir="output_fp32",
    overwrite_output_dir=True,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
    fp16=False
)

training_args_fp16 = TrainingArguments(
    output_dir="output_fp16",
    overwrite_output_dir=True,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
    fp16=torch.cuda.is_available()
)

trainer_fp32 = Trainer(
    model=model_fp32,
    args=training_args_fp32,
    train_dataset=small_train,
    eval_dataset=small_test
)

trainer_fp16 = Trainer(
    model=model_fp16,
    args=training_args_fp16,
    train_dataset=small_train,
    eval_dataset=small_test
)

#%% --------------------------------------------------------------------------------------------------------------------
print("----- Training in FP32 -----")
trainer_fp32.train()
results_fp32 = trainer_fp32.evaluate()
print("FP32 results:", results_fp32)

print("----- Training in FP16 -----")
trainer_fp16.train()
results_fp16 = trainer_fp16.evaluate()
print("FP16 results:", results_fp16)
