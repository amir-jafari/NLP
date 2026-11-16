import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation
'''
QLoRA = Quantized base model + Low-Rank Adapters
  1) Load the LLM with its weights compressed to 4 bits (bitsandbytes)  -> ~4x less GPU memory
  2) Freeze it, and add small trainable LoRA matrices to the attention/MLP layers (peft)
  3) Train ONLY the adapters (a few MB) on instruction data (Dolly format)
  pip install bitsandbytes peft accelerate
'''
load_dotenv()
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"../config/{os.environ['CONFIG_FILE']}")
cfg = config['casualLM_QLoRA']

SYSTEM_PROMPT = "You are a helpful assistant. Answer the instruction, using the context if one is given."


# ---------------------------------------------------------------------------------------------------------------------
# Model loading
def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_4bit_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # store weights in 4 bits
        bnb_4bit_quant_type="nf4",              # "NormalFloat4": best 4-bit format for normally distributed weights
        bnb_4bit_use_double_quant=True,         # also quantize the quantization constants (saves a bit more memory)
        bnb_4bit_compute_dtype=torch.bfloat16,  # de-quantize to bf16 for the actual matrix multiplications
    )
    return AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": 0})


def add_lora_adapters(model):
    model = prepare_model_for_kbit_training(model)    # freeze base weights, enable gradient checkpointing
    lora_config = LoraConfig(
        r=int(cfg['lora_r']),                      # rank of the small update matrices A (d x r) and B (r x d)
        lora_alpha=int(cfg['lora_alpha']),         # scaling of the update
        lora_dropout=float(cfg['lora_dropout']),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------------------------------------------------
# Data: Dolly rows (instruction, context, response) -> chat messages -> token ids + labels
def build_messages(instruction, context=None):
    user = instruction if not context or pd.isna(context) else f"{instruction}\n\nContext:\n{context}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def tokenize_example(rec, tokenizer, max_length):
    prompt = tokenizer.apply_chat_template(build_messages(rec["instruction"], rec["context"]),
                                           tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(str(rec["response"]) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    # Loss only on the answer: prompt tokens get label -100 (ignored by the loss)
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def load_training_dataset(tokenizer):
    df = pd.read_csv(cfg['file_path'])
    dataset = Dataset.from_pandas(df)
    return dataset.map(lambda rec: tokenize_example(rec, tokenizer, int(cfg['max_length'])),
                       remove_columns=dataset.column_names)


# ---------------------------------------------------------------------------------------------------------------------
# Inference
def generate(model, tokenizer, instruction, context=None, max_new_tokens=150):
    prompt = tokenizer.apply_chat_template(build_messages(instruction, context), tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
