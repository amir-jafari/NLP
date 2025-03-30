#%% --------------------------------------------------------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,
    )

#%% --------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2"

#%% --------------------------------------------------------------------------------------------------------------------
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
train_dataset = dataset["train"].shuffle(seed=42)
eval_dataset = dataset["validation"].shuffle(seed=42)

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)

train_dataset = train_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

#%% --------------------------------------------------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(device)

#%% --------------------------------------------------------------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="Lora_out",
    save_strategy="no",
    eval_strategy="epoch",
    logging_steps=5,
    num_train_epochs=1,
    per_device_train_batch_size=32, # ****** the parameter is set for 32GB Memory g5.2xlarge *****
    per_device_eval_batch_size=16, # ****** the parameter is set for 32GB Memory g5.2xlarge *****
    learning_rate=1e-3,
    gradient_accumulation_steps=4,
    fp16=torch.cuda.is_available()
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
model.eval()

#%% --------------------------------------------------------------------------------------------------------------------
test_prompt = "What is Natural Language Processing in the computer science area?"
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("=== GENERATED TEXT ===")
print(generated_text)
