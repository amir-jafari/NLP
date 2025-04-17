# %% ------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim


# %% ------------------------------------------------------------------------------------------------------------
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
        out = self.fc(out)  # Apply linear layer
        return out, hidden  # Return the output and the new hidden state

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
            decoder_input = torch.zeros(input_seq.size(0), 1, self.encoder.hidden_size).to(
                input_seq.device)  # Shape: (batch_size, 1, hidden_size)

        output_seq = torch.cat(outputs, dim=1)  # Concatenate all outputs along time dimension
        return output_seq, encoder_hidden      # Return the output sequence and encoder's final hidden state


# %% ------------------------------------------------------------------------------------------------------------

input_size = 5        # Encoder input dimension
hidden_size = 10      # Shared hidden dimension for encoder/decoder RNNs
output_size = 5       # Decoder output dimension (e.g., size of target vocabulary)
seq_len = 7           # Length of the input sequence
target_len = 7        # Length of the target sequence
batch_size = 3        # Number of sequences in a batch
learning_rate = 0.001
num_epochs = 100     # Number of training epochs
# %% ------------------------------------------------------------------------------------------------------------
encoder = EncoderRNN(input_size, hidden_size)
decoder = DecoderRNN(hidden_size, output_size)
seq2seq_model = Seq2Seq(encoder, decoder, target_len)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(seq2seq_model.parameters(), lr=learning_rate)

# %% ------------------------------------------------------------------------------------------------------------
input_seq = torch.randn(batch_size, seq_len, input_size)       # Shape: (3, 7, 5)
target_seq = torch.randn(batch_size, target_len, output_size)  # Shape: (3, 7, 5) for the target

# %% ------------------------------------------------------------------------------------------------------------

for epoch in range(num_epochs):
    seq2seq_model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients
    output_seq, encoder_hidden = seq2seq_model(input_seq)
    loss = criterion(output_seq, target_seq)
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# %% ------------------------------------------------------------------------------------------------------------
print("Final encoder hidden state shape:", encoder_hidden.shape)  # Expected: (1, batch_size, hidden_size)
print("Final decoder output sequence shape:", output_seq.shape)  # Expected: (batch_size, target_len, output_size)

# %% ------------------------------------------------------------------------------------------------------------
