#%% --------------------------------------------------------------------------------------------------------------------
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

#%% --------------------------------------------------------------------------------------------------------------------
data = load_dataset("imdb")
train_data = data["train"].shuffle(seed=42).select(range(2000))
eval_data = data["test"].shuffle(seed=42).select(range(800))

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_data = train_data.map(tokenize, batched=True)
eval_data = eval_data.map(tokenize, batched=True)

#%% --------------------------------------------------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
config = LoraConfig( r=8,lora_alpha=32,target_modules=["query", "value"],lora_dropout=0.05,bias="none", task_type="SEQ_CLS")
model = get_peft_model(model, config)

#%% --------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#%% --------------------------------------------------------------------------------------------------------------------
training_args = TrainingArguments(output_dir="peft_out", num_train_epochs=1, per_device_train_batch_size=4, eval_strategy="epoch")
trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=eval_data)
trainer.train()
print(trainer.evaluate())

#%% --------------------------------------------------------------------------------------------------------------------
test_text = "I absolutely loved this movie."
encoded_input = tokenizer( test_text, return_tensors="pt", truncation=True, max_length=16)
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
with torch.no_grad():
    output = model(**encoded_input)
    prediction = torch.argmax(output.logits, dim=-1).item()

print("Text:", test_text)
print("Predicted label:", prediction)