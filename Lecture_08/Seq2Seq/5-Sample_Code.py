import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
# %% ------------------------------------------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# %% ------------------------------------------------------------------------------------------------------------

SRC_VOCAB_SIZE = 10  # Size of the source vocabulary
TRG_VOCAB_SIZE = 10  # Size of the target vocabulary
EMB_DIM = 8          # Embedding dimension
HIDDEN_DIM = 16      # Hidden dimension size
N_EPOCHS = 100       # Number of training epochs
BATCH_SIZE = 32      # Batch size
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 0.001

# %% ------------------------------------------------------------------------------------------------------------

def generate_dummy_data(num_samples=100, max_length=5):
    src = torch.randint(1, SRC_VOCAB_SIZE, (num_samples, max_length))
    trg = torch.randint(1, TRG_VOCAB_SIZE, (num_samples, max_length))
    return src, trg
# %% ------------------------------------------------------------------------------------------------------------

def create_batches(src_data, trg_data, batch_size):
    num_samples = len(src_data)
    for i in range(0, num_samples, batch_size):
        yield (src_data[i:i + batch_size],
               trg_data[i:i + batch_size])
# %% ------------------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, src_seq_len, hidden_dim)
        # Expand decoder hidden state to match encoder outputs sequence length
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)

        energy = torch.tanh(self.Wa(decoder_hidden) + self.Ua(encoder_outputs))
        scores = self.Va(energy)
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)
        return context_vector, attention_weights
# %% ------------------------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        # src: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch_size, src_len, emb_dim)
        outputs, hidden = self.rnn(embedded)  # outputs: (batch_size, src_len, hidden_dim)
        return outputs, hidden.squeeze(0)
# %% ------------------------------------------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, trg, hidden, encoder_outputs):
        # trg: (batch_size, 1)
        # hidden: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, src_len, hidden_dim)

        embedded = self.dropout(self.embedding(trg))  # (batch_size, 1, emb_dim)
        context_vector, attention_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, context_vector), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output = torch.cat((output, context_vector), dim=2)
        prediction = self.fc_out(output)
        return prediction, hidden.squeeze(0), attention_weights
# %% ------------------------------------------------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        dec_input = trg[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            dec_output, hidden, _ = self.decoder(dec_input, hidden, encoder_outputs)
            outputs[:, t:t + 1] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(2)
            dec_input = trg[:, t].unsqueeze(1) if teacher_force else top1
        return outputs

# %% ------------------------------------------------------------------------------------------------------------

def train_epoch(model, data_iterator, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0

    for src, trg in data_iterator:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss
# %% ------------------------------------------------------------------------------------------------------------
def evaluate(model, data_iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in data_iterator:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # Turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss
# %% ------------------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(SRC_VOCAB_SIZE, EMB_DIM, HIDDEN_DIM).to(device)
decoder = Decoder(TRG_VOCAB_SIZE, EMB_DIM, HIDDEN_DIM).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
train_src, train_trg = generate_dummy_data(num_samples=1000)
valid_src, valid_trg = generate_dummy_data(num_samples=100)
best_valid_loss = float('inf')
# %% ------------------------------------------------------------------------------------------------------------

for epoch in range(N_EPOCHS):
    train_iterator = create_batches(train_src, train_trg, BATCH_SIZE)
    valid_iterator = create_batches(valid_src, valid_trg, BATCH_SIZE)
    train_loss = train_epoch(model, train_iterator, optimizer, criterion)
    valid_loss = evaluate(model, valid_iterator, criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')

model.eval()
test_src, test_trg = generate_dummy_data(num_samples=5)
test_src, test_trg = test_src.to(device), test_trg.to(device)
# %% ------------------------------------------------------------------------------------------------------------
with torch.no_grad():
    output = model(test_src, test_trg, 0)  # Turn off teacher forcing
    predictions = output.argmax(dim=-1)

    for i in range(5):
        print(f'\nSource sequence: {test_src[i].cpu().numpy()}')
        print(f'Target sequence: {test_trg[i].cpu().numpy()}')
        print(f'Predicted sequence: {predictions[i].cpu().numpy()}')