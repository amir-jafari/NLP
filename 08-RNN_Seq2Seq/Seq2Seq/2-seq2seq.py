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
        outputs, hidden = self.rnn(x)  # outputs: (batch_size, seq_len, hidden_size)
        return outputs, hidden  # Return all outputs and the last hidden state

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)  # Expecting input x of shape (batch_size, seq_len=1, hidden_size)
        out = self.fc(out)
        return out, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_len):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_len = target_len

    def forward(self, input_seq):
        encoder_outputs, encoder_hidden = self.encoder(input_seq)  # (batch_size, seq_len, hidden_size)
        decoder_input = torch.zeros(input_seq.size(0), 1, self.encoder.hidden_size).to(input_seq.device)
        decoder_hidden = encoder_hidden  # Initialize decoder hidden state with encoder's hidden state

        outputs = []
        for t in range(self.target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs.append(decoder_output)  # Append the output for this time step
            decoder_input = torch.zeros(input_seq.size(0), 1, self.encoder.hidden_size).to(input_seq.device)  # Shape: (batch_size, 1, hidden_size)

        output_seq = torch.cat(outputs, dim=1)  # Concatenate all outputs along time dimension
        return output_seq, encoder_hidden  # Return the output sequence and encoder's final hidden state

#%% ------------------------------------------------------------------------------------------------------------
input_size = 5    # Encoder input dimension
hidden_size = 10  # Shared hidden dimension for encoder/decoder RNNs
output_size = 5   # Decoder output dimension (e.g., size of target vocabulary)
seq_len = 7       # Length of the input sequence
target_len = 7    # Length of the target sequence
batch_size = 3    # Number of sequences in a batch

encoder = EncoderRNN(input_size, hidden_size)
decoder = DecoderRNN(hidden_size, output_size)
seq2seq_model = Seq2Seq(encoder, decoder, target_len)

#%% ------------------------------------------------------------------------------------------------------------
input_seq = torch.randn(batch_size, seq_len, input_size)  # Shape: (3, 7, 5)
output_seq, encoder_hidden = seq2seq_model(input_seq)  # Capture the encoder's hidden state

print("Encoder final hidden state shape:", encoder_hidden.shape)  # Expected: (1, batch_size, hidden_size)
print("Decoder output sequence shape:", output_seq.shape)  # Expected: (batch_size, target_len, output_size)

#%% ------------------------------------------------------------------------------------------------------------
