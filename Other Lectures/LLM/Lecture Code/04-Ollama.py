#%% --------------------------------------------------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

#%% --------------------------------------------------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

#%% --------------------------------------------------------------------------------------------------------------------
def ollama_parallel_inference(model, texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    attention_mask = inputs['attention_mask']
    outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=100, num_return_sequences=1,
                             temperature=0.7, top_p=0.9)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

#%% --------------------------------------------------------------------------------------------------------------------
input_texts = ["Explain AI."]
results = ollama_parallel_inference(model, input_texts)
for res in results:
    print(res)