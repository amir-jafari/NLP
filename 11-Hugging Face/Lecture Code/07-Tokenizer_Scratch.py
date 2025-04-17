#%% --------------------------------------------------------------------------------------------------------------------
from tokenizers import ByteLevelBPETokenizer
import pandas as pd

#%% --------------------------------------------------------------------------------------------------------------------
dataset = pd.read_csv('20newsgroups.txt', sep='\t')
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(dataset['text'], vocab_size=2000, min_frequency=2)
vocab = tokenizer.get_vocab()

#%% --------------------------------------------------------------------------------------------------------------------
print("First 20 tokens and their IDs:")
for token, idx in list(vocab.items())[:20]:
    print(f"{token}: {idx}")
