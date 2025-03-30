#%% --------------------------------------------------------------------------------------------------------------------
# packages
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

#%% --------------------------------------------------------------------------------------------------------------------
# model and QLoRA config
model_name = "gpt2-medium"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#%% --------------------------------------------------------------------------------------------------------------------
# load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset["train"].shuffle(seed=42).select(range(1000))
eval_data = dataset["validation"].shuffle(seed=42).select(range(200))

#%% --------------------------------------------------------------------------------------------------------------------
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64)

train_data = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
eval_data = eval_data.map(tokenize_function, batched=True, remove_columns=["text"])

train_data = train_data.filter(lambda x: len(x["input_ids"]) > 0)
eval_data = eval_data.filter(lambda x: len(x["input_ids"]) > 0)

train_data.set_format("torch")
eval_data.set_format("torch")

#%% --------------------------------------------------------------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="lora_4bit_out",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=5,
    save_strategy="no",
    learning_rate=1e-4,
    gradient_accumulation_steps=2
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=eval_data
)

#%% --------------------------------------------------------------------------------------------------------------------
# train
trainer.train()
model.eval()

#%% --------------------------------------------------------------------------------------------------------------------
test_prompt = "What is Natural Language Processing?"
inputs = tokenizer(test_prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
