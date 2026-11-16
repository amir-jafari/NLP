import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
'''
Quick test: run an instruction-tuned LLM that is quantized to 4 bits on the fly (bitsandbytes).
4-bit weights need ~4x less GPU memory than 16-bit, so much larger models fit on one GPU.
  pip install bitsandbytes accelerate
'''
model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"       # try "Qwen/Qwen2.5-7B-Instruct" on a big GPU

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16),
    device_map={"": 0} if torch.cuda.is_available() else None,
)
print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.0f} MB")

prompt = "Tell me about AI"
messages = [{"role": "user", "content": prompt}]
prompt_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n\n*** Generate:")
inputs = tokenizer(prompt_template, return_tensors='pt').to(model.device)
output = model.generate(**inputs, do_sample=True, temperature=0.7, max_new_tokens=200)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# Inference can also be done using transformers' pipeline
print("\n*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)
print(pipe(messages)[0]['generated_text'][-1]['content'])
