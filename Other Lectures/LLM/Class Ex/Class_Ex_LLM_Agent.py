# ======================================================================================================================
# Class_Ex_LoRA_QLoRA.py
# ======================================================================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

# ======================================================================================================================
# Class_Ex1:
# How can you apply LoRA (Low-Rank Adaptation) to a smaller model, such as "distilgpt2", and generate text?
# Step 1: Load "distilgpt2" locally with AutoModelForCausalLM.
# Step 2: Use a LoraConfig with r=4, lora_alpha=16, and lora_dropout=0.1 (or other hyperparams).
# Step 3: Call get_peft_model(...) and then generate output for a sample prompt.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')



print(20*'-' + 'End Q1' + 20*'-')


# ======================================================================================================================
# Class_Ex2:
# How can you measure the number of trainable parameters in a LoRA-injected model versus the original?
# Step 1: Load a base model, count trainable parameters using sum(p.numel() for p in model.parameters() if p.requires_grad)
# Step 2: Inject LoRA, count again, and compare the difference.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')



print(20*'-' + 'End Q2' + 20*'-')


# ======================================================================================================================
# Class_Ex3:
# How can you apply QLoRA (4-bit quantization + LoRA) to reduce GPU memory usage?
# Step 1: Install bitsandbytes if not already (pip install bitsandbytes).
# Step 2: Load a base model with load_in_4bit=True and device_map="auto".
# Step 3: Apply the same LoRA config, generate text, and note memory usage (e.g. via nvidia-smi).
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')



print(20*'-' + 'End Q3' + 20*'-')


# ======================================================================================================================
# Class_Ex4:
# When should you prefer LoRA vs. QLoRA?
# Step 1: Create a short textual explanation comparing memory usage, performance, and ease of setup.
# Step 2: Print or log your reasoning in the console.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')



print(20*'-' + 'End Q4' + 20*'-')


# ======================================================================================================================
# Class_Ex5:
# How can you build a simple “multi-agent” pipeline where:
#  - Agent A: Writes a short piece of text
#  - Agent B: Edits it for style
#  - Agent C: Fact-checks by rewriting or printing a "verified" statement
# Step 1: Create three separate prompts or “roles” for your LLM, each representing an agent.
# Step 2: Pass the output from one agent as input to the next.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')



print(20*'-' + 'End Q5' + 20*'-')
