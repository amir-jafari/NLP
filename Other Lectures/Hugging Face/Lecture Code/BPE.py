#%% --------------------------------------------------------------------------------------------------------------------
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer_bpe = Tokenizer(BPE())
tokenizer_bpe.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=3000, show_progress=True, initial_alphabet=[])
texts = ["Hello world!", "Byte-Pair Encoding merges frequent character sequences."]

#%% --------------------------------------------------------------------------------------------------------------------
tokenizer_bpe.train_from_iterator(texts, trainer=trainer)
encoded_bpe = tokenizer_bpe.encode("Hello world!")
print(encoded_bpe.tokens)
