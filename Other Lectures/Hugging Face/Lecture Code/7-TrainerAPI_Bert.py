#%% --------------------------------------------------------------------------------------------------------------------
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
#%% --------------------------------------------------------------------------------------------------------------------
dataset = load_dataset("ag_news")
train_ds = dataset["train"].shuffle(seed=42).select(range(500))
val_ds = dataset["test"].shuffle(seed=42).select(range(100))

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=4)

#%% --------------------------------------------------------------------------------------------------------------------
train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format('torch', columns=columns)
val_ds.set_format('torch', columns=columns)

#%% --------------------------------------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    logging_steps=10,
    logging_strategy='steps',
    disable_tqdm=False
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
