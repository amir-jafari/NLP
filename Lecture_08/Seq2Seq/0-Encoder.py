import torch
import torch.nn as nn
#%% ------------------------------------------------------------------------------------------------------------
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out)
        return out
#%% ------------------------------------------------------------------------------------------------------------
input_size = 5
hidden_size = 10
output_size = 5
seq_len = 7
batch_size = 3
#%% ------------------------------------------------------------------------------------------------------------
model = SimpleRNN(input_size, hidden_size, output_size)
input_seq = torch.randn(batch_size, seq_len, input_size)
output_seq = model(input_seq)

print("Input sequence shape:", input_seq.shape)
print("Output sequence shape:", output_seq.shape)
