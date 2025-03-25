#%% --------------------------------------------------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model

def tokenize_function(examples):
    return tokenizer( examples["text"], padding="max_length", truncation=True, max_length=16)

def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    preds = logits.argmax(dim=-1)
    accuracy = (preds == labels).float().mean()
    return {"accuracy": accuracy.item()}

#%% --------------------------------------------------------------------------------------------------------------------
data = {
    "text": [ "I love puppies", "I hate this", "I adore cats", "I dislike chores", "I cherish sunshine", "I detest cold weather"],
    "label": [1, 0, 1, 0, 1, 0]
}

MODEL_NAME = "bert-base-uncased"
dataset = HFDataset.from_dict(data)
train_dataset = dataset.select(range(4))
eval_dataset = dataset.select(range(4, 6))

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

#%% --------------------------------------------------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
peft_config = LoraConfig( r=8,lora_alpha=32,target_modules=["query", "value"],lora_dropout=0.05,bias="none", task_type="SEQ_CLS")

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
peft_model.to(device)

#%% --------------------------------------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="peft-lora-bert",
    eval_strategy="steps",
    eval_steps=10,
    save_steps=10,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=5,
    learning_rate=1e-4
)

trainer = Trainer(model=peft_model, args=training_args,train_dataset=train_dataset, eval_dataset=eval_dataset,
                  compute_metrics=compute_metrics)
trainer.train()
peft_model.eval()

#%% --------------------------------------------------------------------------------------------------------------------
test_text = "I really love playing with dogs."
encoded_input = tokenizer( test_text, return_tensors="pt", truncation=True, max_length=16)
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

with torch.no_grad():
    output = peft_model(**encoded_input)
    prediction = torch.argmax(output.logits, dim=-1).item()

print("Text:", test_text)
print("Predicted label:", prediction)
