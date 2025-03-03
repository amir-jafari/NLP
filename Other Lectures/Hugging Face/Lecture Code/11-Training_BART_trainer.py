#%% --------------------------------------------------------------------------------------------------------------------
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch

def preprocess(examples):
    inputs = tokenizer(examples['text'], max_length=64, truncation=True)
    labels = tokenizer(examples['text'], max_length=32, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

#%% --------------------------------------------------------------------------------------------------------------------
dataset = load_dataset("ag_news", split="train[:200]").shuffle(seed=42)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenized_ds = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest')

#%% --------------------------------------------------------------------------------------------------------------------
args = TrainingArguments(output_dir='bart-sum', num_train_epochs=1, per_device_train_batch_size=32, logging_steps=5,fp16=torch.cuda.is_available(),)
trainer = Trainer(model=model, args=args, train_dataset=tokenized_ds, data_collator=data_collator)
trainer.train()
