#%% --------------------------------------------------------------------------------------------------------------------
from transformers import MarianMTModel, MarianTokenizer

#%% --------------------------------------------------------------------------------------------------------------------
src_lang = 'en'
tgt_lang = 'fr'
model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

#%% --------------------------------------------------------------------------------------------------------------------
text = 'Hello world!'
encoded = tokenizer([text], return_tensors='pt', padding=True)
translated = model.generate(**encoded)
translation = tokenizer.decode(translated[0], skip_special_tokens=True)
print(translation)
