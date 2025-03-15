#%% --------------------------------------------------------------------------------------------------------------------
from transformers import BartTokenizer, BartForConditionalGeneration

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

#%% --------------------------------------------------------------------------------------------------------------------
input_text = 'Bart is a powerful seq2seq model developed by Facebook AI.'
inputs = tokenizer.encode(input_text, return_tensors='pt')

#%% --------------------------------------------------------------------------------------------------------------------
summary_ids = model.generate(inputs, max_length=30, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
