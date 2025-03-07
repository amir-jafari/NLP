#%% --------------------------------------------------------------------------------------------------------------------
from transformers import BertTokenizerFast

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer_wordpiece = BertTokenizerFast.from_pretrained("bert-base-uncased")
text = "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language efficiently."

#%% --------------------------------------------------------------------------------------------------------------------
encoded_wordpiece = tokenizer_wordpiece.encode_plus(text, add_special_tokens=True, return_tensors="pt")
print("Tokenizer Method: WordPiece (BERT)")
print("Text:           ", text)
print("Tokens:         ", tokenizer_wordpiece.convert_ids_to_tokens(encoded_wordpiece["input_ids"][0]))
print("IDs:            ", encoded_wordpiece["input_ids"].tolist()[0])
print("Decoded Text:   ", tokenizer_wordpiece.decode(encoded_wordpiece["input_ids"][0]))