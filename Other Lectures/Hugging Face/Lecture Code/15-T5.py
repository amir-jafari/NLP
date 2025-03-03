#%% --------------------------------------------------------------------------------------------------------------------
from transformers import T5Tokenizer, T5ForConditionalGeneration

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

#%% --------------------------------------------------------------------------------------------------------------------
input_text = 'summarize: Encoder-decoder models are widely used.'
input_ids = tokenizer.encode(input_text, return_tensors='pt')

#%% --------------------------------------------------------------------------------------------------------------------
summary_ids = model.generate(input_ids, max_length=30, num_beams=2)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
