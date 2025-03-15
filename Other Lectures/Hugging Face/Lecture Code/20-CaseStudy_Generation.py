#%% --------------------------------------------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM

#%% --------------------------------------------------------------------------------------------------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#%% --------------------------------------------------------------------------------------------------------------------
prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs, 
    max_length=50, 
    do_sample=True, 
    temperature=0.7
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)