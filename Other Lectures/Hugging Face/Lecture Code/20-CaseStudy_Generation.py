#%% --------------------------------------------------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt: str, max_length: int = 50) -> str:
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

#%% --------------------------------------------------------------------------------------------------------------------
prompt_text = "Once upon a time,"
generated = generate_text(prompt_text, max_length=50)
print(f"Prompt: {prompt_text}")
print("Generated Text:")
print(generated)