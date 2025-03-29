#%% --------------------------------------------------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

#%% --------------------------------------------------------------------------------------------------------------------
def compare_fp16_fp32(model_name="gpt2", prompt="Hello, world!"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")

    # ---------------- FP32 ----------------
    model_fp32 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    start_fp32 = time.time()
    outputs_fp32 = model_fp32.generate(**inputs, max_new_tokens=30)
    end_fp32 = time.time()
    text_fp32 = tokenizer.decode(outputs_fp32[0], skip_special_tokens=True)

    # ---------------- FP16 ----------------
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    inputs_fp16 = inputs.to("cuda")
    start_fp16 = time.time()
    outputs_fp16 = model_fp16.generate(**inputs_fp16, max_new_tokens=30)
    end_fp16 = time.time()
    text_fp16 = tokenizer.decode(outputs_fp16[0], skip_special_tokens=True)

    print("=== FP32 Output ===")
    print(text_fp32)
    print(f"Time Elapsed (FP32): {end_fp32 - start_fp32:.2f} seconds\n")

    print("=== FP16 Output ===")
    print(text_fp16)
    print(f"Time Elapsed (FP16): {end_fp16 - start_fp16:.2f} seconds\n")

#%% --------------------------------------------------------------------------------------------------------------------
compare_fp16_fp32("gpt2", "Hello, I'm testing FP precision.")
