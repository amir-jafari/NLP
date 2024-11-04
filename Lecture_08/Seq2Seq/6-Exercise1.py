"""
Neural Machine Translation Exercise Template
Students should complete all TODO sections
"""
# %% ------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from collections import Counter
# %% ------------------------------------------------------------------------------------------------------------
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# %% ------------------------------------------------------------------------------------------------------------

# Sample parallel corpus (English-Spanish pairs)
parallel_corpus = [
    ("hello", "hola"),
    ("thank you", "gracias"),
    ("good morning", "buenos días"),
    # Add more pairs as needed
]
# %% ------------------------------------------------------------------------------------------------------------

class Vocabulary:
    def __init__(self):
        # TODO: Initialize vocabulary with special tokens
        # Hint: Need <pad>, <sos>, <eos>, <unk>
        pass

    def build_vocabulary(self, sentences):
        """Build vocabulary from list of sentences"""
        # TODO: Implement vocabulary building
        # 1. Count word frequencies
        # 2. Add words to vocabulary above minimum frequency
        pass

    def sentence_to_indices(self, sentence, max_length=10):
        """Convert sentence to indices with padding"""
        # TODO: Implement conversion of sentence to indices
        # 1. Split sentence into words
        # 2. Convert words to indices
        # 3. Add <sos> and <eos> tokens
        # 4. Handle padding
        pass

    def indices_to_sentence(self, indices):
        """Convert indices back to sentence"""
        # TODO: Implement conversion of indices to sentence
        # 1. Convert indices to words
        # 2. Handle special tokens
        # 3. Join words into sentence
        pass
# %% ------------------------------------------------------------------------------------------------------------

class TranslationDataset(Dataset):
    def __init__(self, corpus, src_vocab, trg_vocab, max_length=10):
        # TODO: Initialize dataset
        pass

    def __len__(self):
        # TODO: Return length of dataset
        pass

    def __getitem__(self, idx):
        # TODO: Get item from dataset
        # 1. Get source and target sentences
        # 2. Convert to indices
        # 3. Return as tensors
        pass
# %% ------------------------------------------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # TODO: Initialize attention layers
        pass

    def forward(self, decoder_hidden, encoder_outputs):
        # TODO: Implement attention mechanism
        # 1. Calculate attention scores
        # 2. Apply softmax to get attention weights
        # 3. Create and return context vector
        pass
# %% ------------------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        # TODO: Initialize encoder layers
        pass

    def forward(self, src):
        # TODO: Implement encoder forward pass
        # 1. Embed input
        # 2. Pass through RNN
        # 3. Process hidden state
        pass
# %% ------------------------------------------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        # TODO: Initialize decoder layers
        pass

    def forward(self, trg, hidden, encoder_outputs):
        # TODO: Implement decoder forward pass
        # 1. Embed input
        # 2. Calculate attention
        # 3. Create RNN input
        # 4. Generate output
        pass
# %% ------------------------------------------------------------------------------------------------------------

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        # TODO: Initialize Seq2Seq model
        pass

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # TODO: Implement Seq2Seq forward pass
        # 1. Get batch size and target length
        # 2. Initialize outputs tensor
        # 3. Get encoder outputs
        # 4. Teacher forcing loop
        pass
# %% ------------------------------------------------------------------------------------------------------------

def train(model, iterator, optimizer, criterion, clip, device):
    # TODO: Implement training loop
    # 1. Set model to training mode
    # 2. Initialize loss
    # 3. Process batches
    # 4. Implement gradient clipping
    pass
# %% ------------------------------------------------------------------------------------------------------------

def evaluate(model, iterator, criterion, device):
    # TODO: Implement evaluation loop
    # 1. Set model to evaluation mode
    # 2. Disable gradient calculation
    # 3. Calculate loss on batches
    pass
# %% ------------------------------------------------------------------------------------------------------------

def translate_sentence(model, sentence, src_vocab, trg_vocab, device, max_length=10):
    # TODO: Implement translation function
    # 1. Prepare input sentence
    # 2. Get encoder outputs
    # 3. Generate translation
    # 4. Convert output to sentence
    pass
# %% ------------------------------------------------------------------------------------------------------------

def main():
    # Hyperparameters
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    DROPOUT = 0.5
    N_EPOCHS = 100
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    CLIP = 1.0
    MAX_LENGTH = 15

    # TODO: Implement main training loop
    # 1. Setup device
    # 2. Create vocabularies
    # 3. Create dataset and iterator
    # 4. Initialize model
    # 5. Train model
    # 6. Test translations

    # Test sentences
    test_sentences = [
        "hello",
        "thank you",
        "how are you",
        "good morning",
        "i love programming"
    ]

if __name__ == "__main__":
    main()