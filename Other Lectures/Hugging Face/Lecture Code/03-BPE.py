#%% --------------------------------------------------------------------------------------------------------------------
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer_bpe = Tokenizer(BPE())
tokenizer_bpe.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=3000, show_progress=True, initial_alphabet=[])
texts = ["Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language efficiently.", "Byte-Pair Encoding merges frequent character sequences and optimizes vocabulary size."]
text = "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language efficiently."

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer_bpe.train_from_iterator(texts, trainer=trainer)
encoded_bpe = tokenizer_bpe.encode(text)

#%% --------------------------------------------------------------------------------------------------------------------
print("Tokenizer Method: BPE")
print("Text:           ", text)
print("Tokens:         ", encoded_bpe.tokens)
print("IDs:            ", encoded_bpe.ids)
print("Decoded Text:   ", tokenizer_bpe.decode(encoded_bpe.ids))
