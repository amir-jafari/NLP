#%% --------------------------------------------------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

# %% --------------------------------------------------------------------------
model_name = "EleutherAI/gpt-neo-1.3B"
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,inference_mode=True,r=4,lora_alpha=16,lora_dropout=0.1)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = get_peft_model(model, peft_config)
print("=== Trainable Parameters (LoRA) ===")
model.print_trainable_parameters()

# %% --------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "Natural Language Processing is"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# %% --------------------------------------------------------------------------
print("=== LoRA Results ===")
print("Prompt:", prompt)
print("Generated:", generated_text)
