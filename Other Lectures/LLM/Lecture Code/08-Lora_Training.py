#%% --------------------------------------------------------------------------------------------------------------------
import torch
from datasets import Dataset
from transformers import (AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64)

#%% --------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "EleutherAI/gpt-neo-1.3B"
train_texts = ["The cat sat on the mat.", "Once upon a time, there was a brave knight.",
               "Artificial intelligence is fascinating.","Large language models can produce human-like text.",]
dataset = Dataset.from_dict({"text": train_texts})
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

#%% --------------------------------------------------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)

#%% --------------------------------------------------------------------------------------------------------------------
lora_config = LoraConfig(r=8,lora_alpha=32,target_modules=["q_proj", "k_proj", "v_proj"],lora_dropout=0.05,bias="none",
                         task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(device)

#%% --------------------------------------------------------------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
training_args = TrainingArguments(output_dir="dummy_dir",save_strategy='no', eval_strategy="no",eval_steps=10,save_steps=10,
                                  logging_steps=5,num_train_epochs=3,per_device_train_batch_size=4,
                                  per_device_eval_batch_size=4,learning_rate=1e-4,gradient_accumulation_steps=4,
                                  fp16=torch.cuda.is_available(),)
trainer = Trainer(model=model,args=training_args,data_collator=data_collator,train_dataset=train_dataset,
    eval_dataset=eval_dataset)
trainer.train()
model.eval()
test_prompt = "Once upon a time"
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

#%% --------------------------------------------------------------------------------------------------------------------
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=30,do_sample=True,top_p=0.9,temperature=0.8)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("=== GENERATED TEXT ===")
print(generated_text)
