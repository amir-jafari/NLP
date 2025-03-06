#%% --------------------------------------------------------------------------------------------------------------------
from transformers import BertTokenizerFast

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer_wordpiece = BertTokenizerFast.from_pretrained("bert-base-uncased")
text = "Hello world!"

#%% --------------------------------------------------------------------------------------------------------------------
encoded_wordpiece = tokenizer_wordpiece.encode_plus(text, add_special_tokens=True, return_tensors="pt")
print(tokenizer_wordpiece.convert_ids_to_tokens(encoded_wordpiece["input_ids"][0]))