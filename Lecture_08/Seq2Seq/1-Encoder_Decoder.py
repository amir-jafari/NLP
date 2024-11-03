#%% ------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
#%% ------------------------------------------------------------------------------------------------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        outputs, hidden = self.rnn(x)
        return outputs, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
#%% ------------------------------------------------------------------------------------------------------------
input_size = 5    # Encoder input dimension
hidden_size = 10  # Shared hidden dimension for encoder/decoder RNNs
output_size = 5   # Decoder output dimension (e.g., size of target vocabulary)
seq_len = 7       # Length of the input sequence
target_len = 7    # Length of the target sequence
batch_size = 3    # Number of sequences in a batch
#%% ------------------------------------------------------------------------------------------------------------
encoder = EncoderRNN(input_size, hidden_size)
decoder = DecoderRNN(hidden_size, output_size)
#%% ------------------------------------------------------------------------------------------------------------

input_seq = torch.randn(batch_size, seq_len, input_size)  # Shape: (batch_size, seq_len, input_size)
encoder_outputs, encoder_hidden = encoder(input_seq)

print("Encoder outputs at each time step:")
for t in range(seq_len):
    print(f"Time step {t + 1}:")
    print(encoder_outputs[:, t, :])  # Output of shape (batch_size, hidden_size) for each time step

#%% ------------------------------------------------------------------------------------------------------------
decoder_input = torch.zeros(batch_size, 1, hidden_size)  # Shape: (batch_size, 1, hidden_size)
outputs = []
decoder_hidden = encoder_hidden  # Start decoding with the encoder's final hidden state
for t in range(target_len):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    outputs.append(decoder_output)
    decoder_input = torch.zeros(batch_size, 1, hidden_size)  # Reset for each step

output_seq = torch.cat(outputs, dim=1)
#%% ------------------------------------------------------------------------------------------------------------
print("\nEncoder final hidden state shape:", encoder_hidden.shape)  # Expected: (1, batch_size, hidden_size)
print("Decoder output sequence shape:", output_seq.shape)  # Expected: (batch_size, target_len, output_size)
