#%% --------------------------------------------------------------------------------------------------------------------
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

def preprocess(ex):
    inputs = tokenizer(ex['text'], max_length=512, truncation=True)
    summary = tokenizer(ex['text'], max_length=128, truncation=True)
    inputs["labels"] = summary["input_ids"]
    return inputs

#%% --------------------------------------------------------------------------------------------------------------------
dataset = load_dataset("ag_news")
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

#%% --------------------------------------------------------------------------------------------------------------------
train_ds = dataset['train'].map(preprocess, batched=True)
train_ds.set_format('torch', columns=['input_ids','attention_mask','labels'])

#%% --------------------------------------------------------------------------------------------------------------------
args = TrainingArguments(output_dir='bart-sum', num_train_epochs=1)
trainer = Trainer(model=model, args=args, train_dataset=train_ds)
trainer.train()
