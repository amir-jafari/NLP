#%% --------------------------------------------------------------------------------------------------------------------
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

#%% --------------------------------------------------------------------------------------------------------------------
prompt = '13-Transformers are'
input_ids = tokenizer.encode(prompt, return_tensors='pt')

#%% --------------------------------------------------------------------------------------------------------------------
output_ids = model.generate(input_ids, max_length=30, num_beams=2, no_repeat_ngram_size=2)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)
