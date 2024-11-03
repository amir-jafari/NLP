# %% ------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
# %% ------------------------------------------------------------------------------------------------------------

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)   # Weight for decoder hidden state
        self.Ua = nn.Linear(hidden_size, hidden_size)   # Weight for encoder outputs
        self.Va = nn.Linear(hidden_size, 1)  # Weight for producing attention scores

    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)   # Shape: (batch_size, 1, hidden_size)
        scores = self.Va(torch.tanh(self.Wa(decoder_hidden_expanded) + self.Ua(encoder_outputs)))  # Shape: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(scores, dim=1)        # Shape: (batch_size, seq_len, 1)
        context_vector = torch.bmm(attention_weights.permute(0, 2, 1), encoder_outputs)  # Shape: (batch_size, 1, hidden_size)
        return context_vector, attention_weights
# %% ------------------------------------------------------------------------------------------------------------
batch_size = 2    # Number of sequences in a batch
seq_len = 5       # Length of the input sequence
hidden_size = 4   # Hidden dimension size
# %% ------------------------------------------------------------------------------------------------------------
encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)  # Shape: (batch_size, seq_len, hidden_size)
decoder_hidden = torch.randn(batch_size, hidden_size)            # Shape: (batch_size, hidden_size)
# %% ------------------------------------------------------------------------------------------------------------
attention = SimpleAttention(hidden_size)
context_vector, attention_weights = attention(decoder_hidden, encoder_outputs)
# %% ------------------------------------------------------------------------------------------------------------
print("Encoder Outputs:\n", encoder_outputs)
print("\nDecoder Hidden State:\n", decoder_hidden)
print("\nAttention Weights:\n", attention_weights.squeeze())  # Shape: (batch_size, seq_len)
print("\nContext Vector:\n", context_vector)                  # Shape: (batch_size, 1, hidden_size)
