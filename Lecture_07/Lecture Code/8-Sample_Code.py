import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time
import os
import csv

# Define a custom dataset class
class AGNewsDataset(Dataset):
    def __init__(self, split, transform=None):
        assert split in ['train', 'test']
        self.data = self.load_data(split)
        self.transform = transform

    def load_data(self, split):
        data = []
        csv_file = f'{split}.csv'
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                label, text, _ = row
                try:
                    data.append((int(label), text))
                except:
                    pass

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        return label, text


def label_pipeline(label):
    return label - 1


def text_pipeline(text):
    # You can implement custom text preprocessing here if needed
    # For this example, we'll just split the text into tokens
    tokens = text.split()
    token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    return torch.tensor(token_ids, dtype=torch.int64)




# Create data loaders


# Define your vocabulary if necessary
vocab = {}  # You can build your vocabulary here if needed
# Define text and label processing functions
vocab['<unk>'] = len(vocab)  # Add <unk> token

# Create a DataLoader and collate function
def collate_batch(batch):
    labels, texts = zip(*batch)
    labels = torch.tensor([label_pipeline(label) for label in labels], dtype=torch.int64)
    texts = [text_pipeline(text) for text in texts]
    offsets = [0] + [len(text) for text in texts[:-1]]
    offsets = torch.tensor(offsets, dtype=torch.int64).cumsum(dim=0)
    texts = torch.cat(texts)
    return labels.to(device), texts.to(device), offsets.to(device)

# Define the TextClassificationModel
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


EPOCHS = 100
LR = 0.001
BATCH_SIZE = 64
emsize = 128

# Load the AG_NEWS dataset
train_dataset = AGNewsDataset(split='train', transform=text_pipeline)
test_dataset = AGNewsDataset(split='test', transform=text_pipeline)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_class = len(set([label for (label,_,_) in train_dataloader]))
vocab_size = len(vocab)

model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

# Define the training and evaluation functions
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}'.format(epoch, idx, len(dataloader), total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

# Set training parameters

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(test_dataloader)
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} '.format(epoch, time.time() - epoch_start_time, accu_val))
    print('-' * 59)

# Save the trained model
torch.save(model.state_dict(), 'model_weights.pt')
# Load the model for inference
model.load_state_dict(torch.load('model_weights.pt'))

