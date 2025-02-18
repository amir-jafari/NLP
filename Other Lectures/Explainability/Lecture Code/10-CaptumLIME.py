#%%
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

from collections import Counter

from IPython.core.display import HTML, display
#%%
ag_ds = list(AG_NEWS(split='train'))

ag_train, ag_val = ag_ds[:100000], ag_ds[100000:]

tokenizer = get_tokenizer('basic_english')
word_counter = Counter()
for (label, line) in ag_train:
    word_counter.update(tokenizer(line))
voc = Vocab(word_counter, min_freq=10)

print('Vocabulary size:', len(voc))

num_class = len(set(label for label, _ in ag_train))
print('Num of classes:', num_class)

#%%
class EmbeddingBagModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, num_class)

    def forward(self, inputs, offsets):
        embedded = self.embedding(inputs, offsets)
        return self.linear(embedded)


#%%
BATCH_SIZE = 64

def collate_batch(batch):
    labels = torch.tensor([label - 1 for label, _ in batch])
    text_list = [tokenizer(line) for _, line in batch]

    # flatten tokens across the whole batch
    text = torch.tensor([voc[t] for tokens in text_list for t in tokens])
    # the offset of each example
    offsets = torch.tensor(
        [0] + [len(tokens) for tokens in text_list][:-1]
    ).cumsum(dim=0)

    return labels, text, offsets

train_loader = DataLoader(ag_train, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(ag_val, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_batch)

#%%
EPOCHS = 7
EMB_SIZE = 64
CHECKPOINT = './models/embedding_bag_ag_news.pt'
USE_PRETRAINED = True  # change to False if you want to retrain your own model


def train_model(train_loader, val_loader):
    model = EmbeddingBagModel(len(voc), EMB_SIZE, num_class)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, EPOCHS + 1):
        # training
        model.train()
        total_acc, total_count = 0, 0

        for idx, (label, text, offsets) in enumerate(train_loader):
            optimizer.zero_grad()
            predited_label = model(text, offsets)
            loss(predited_label, label).backward()
            optimizer.step()
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

            if (idx + 1) % 500 == 0:
                print('epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}'.format(
                    epoch, idx + 1, len(train_loader), total_acc / total_count
                ))
                total_acc, total_count = 0, 0

                # evaluation
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for label, text, offsets in val_loader:
                predited_label = model(text, offsets)
                total_acc += (predited_label.argmax(1) == label).sum().item()
                total_count += label.size(0)

        print('-' * 59)
        print('end of epoch {:3d} | valid accuracy {:8.3f} '.format(epoch, total_acc / total_count))
        print('-' * 59)

    torch.save(model, CHECKPOINT)
    return model


eb_model = torch.load(CHECKPOINT) if USE_PRETRAINED else train_model(train_loader, val_loader)


#%%
test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
test_line = ('US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
            '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
            'And it is too early to tell if he will match Aleksandr Dityatin, '
            'the Soviet gymnast who won eight total medals in 1980.')

test_labels, test_text, test_offsets = collate_batch([(test_label, test_line)])

probs = F.softmax(eb_model(test_text, test_offsets), dim=1).squeeze(0)
print('Prediction probability:', round(probs[test_labels[0]].item(), 4))