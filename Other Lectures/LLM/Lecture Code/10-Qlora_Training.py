#%% --------------------------------------------------------------------------------------------------------------------
import os
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2-xl"

#%% --------------------------------------------------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
lora_config = LoraConfig(r=64,lora_alpha=16,target_modules=["c_attn", "c_proj"],lora_dropout=0.05,
                         bias="none",task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#%% --------------------------------------------------------------------------------------------------------------------
train_texts = [
    "The cat sat on the mat.",
    "Large language models can produce human-like text.",
    "Once upon a time, a hero rose to the challenge in a faraway land.",
    "Quantization is a great technique to reduce memory footprint."
]

dataset = Dataset.from_dict({"text": train_texts})
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = train_dataset.map(tokenize_function)
eval_dataset = eval_dataset.map(tokenize_function)
train_dataset.set_format("torch")
eval_dataset.set_format("torch")
train_dataset = train_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.remove_columns(["text"])

#%% --------------------------------------------------------------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#%% --------------------------------------------------------------------------------------------------------------------
training_args = TrainingArguments(output_dir="dummy_dir",save_strategy='no', eval_strategy="no",
    per_device_train_batch_size=1,per_device_eval_batch_size=1, logging_steps=5,
    eval_steps=10, save_steps=10, num_train_epochs=1, learning_rate=1e-4,
    fp16=torch.cuda.is_available(), gradient_accumulation_steps=4
)

#%% --------------------------------------------------------------------------------------------------------------------
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                  eval_dataset=eval_dataset, data_collator=data_collator)
trainer.train()

#%% --------------------------------------------------------------------------------------------------------------------
model.eval()
prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True,
                             top_p=0.9, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
