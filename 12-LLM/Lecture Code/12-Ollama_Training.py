#%% --------------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
    category=UserWarning
)

import torch
from datasets import load_dataset
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

#%% --------------------------------------------------------------------------------------------------------------------
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64)

model_name = "openlm-research/open_llama_3b"
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
eval_dataset = dataset["validation"].shuffle(seed=42).select(range(200))

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = LlamaTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
train_dataset = train_dataset.remove_columns(["text"])
train_dataset.set_format("torch")

eval_dataset = eval_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)
eval_dataset = eval_dataset.remove_columns(["text"])
eval_dataset.set_format("torch")

#%% --------------------------------------------------------------------------------------------------------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.eval()

#%% --------------------------------------------------------------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
args = TrainingArguments(
    output_dir="qllama_out",
    save_strategy="no",
    eval_strategy="no",
    logging_steps=5,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    gradient_accumulation_steps=4
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
model.eval()

#%% --------------------------------------------------------------------------------------------------------------------
prompt = "What is Data Visualization?"
inputs = tokenizer(prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))
