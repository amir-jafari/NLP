import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import numpy
# -------------------------------------------------------------------------------------
LR = 1e-2
N_EPOCHS =2000
PRINT_LOSS_EVERY = 1000
EMBEDDING_DIM = 2
# seed = 0
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.cuda.manual_seed(seed)
# numpy.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# os.environ['PYTHONHASHSEED'] = str(seed)
# -------------------------------------------------------------------------------------

corpus = ['king is a strong man',
          'queen is a wise woman',
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']
# -------------------------------------------------------------------------------------
def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))

    return results
# -------------------------------------------------------------------------------------
corpus = remove_stop_words(corpus)
# -------------------------------------------------------------------------------------
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)

words = set(words)
print(words)
# -------------------------------------------------------------------------------------
word2int = {}

for i, word in enumerate(words):
    word2int[word] = i

sentences = []
for sentence in corpus:
    sentences.append(sentence.split())
# -------------------------------------------------------------------------------------
WINDOW_SIZE = 2
data = []

for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])
# -------------------------------------------------------------------------------------
for text in corpus:
    print(text)
df = pd.DataFrame(data, columns = ['input', 'label'])
print(df.head(10))
print(df.shape)
# -------------------------------------------------------------------------------------
ONE_HOT_DIM = len(words)
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding
# -------------------------------------------------------------------------------------
X = []
Y = []
for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))
X_train = np.asarray(X)
Y_train = np.asarray(Y)
# -------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(12, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 12)
        self.act1 = torch.nn.Softmax(dim=1)
    def forward(self, x):
        out_em = self.linear1(x)
        output = self.linear2(out_em)
        output = self.act1(output)
        return out_em, output
p = torch.Tensor(X_train)
p.requires_grad = True
t = torch.Tensor(Y_train)
# %% -------------------------------------- Training Prep --------------------------------------------------------------
model = MLP(EMBEDDING_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
# %% -------------------------------------- Training Loop --------------------------------------------------------------
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    _, t_pred = model(p)
    loss = criterion(t, t_pred)
    loss.backward()
    optimizer.step()
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, loss.item()))

vectors = model.linear1._parameters['weight'].cpu().detach().numpy().transpose()

# s1 = model.linear1._parameters['bias'].cpu().detach().numpy()[0] + model.linear1._parameters['weight'].cpu().detach()[0]
# s2 = model.linear1._parameters['bias'].cpu().detach().numpy()[1] + model.linear1._parameters['weight'].cpu().detach()[1]
# vvectors = torch.cat((s1,s2))
# vectors = vvectors.reshape(12,2).cpu().detach().numpy()
print(vectors)


# -------------------------------------------------------------------------------------
w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
w2v_df['word'] = list(words)
w2v_df = w2v_df[['word', 'x1', 'x2']]
print(w2v_df)
# -------------------------------------------------------------------------------------
fig, ax = plt.subplots()
for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1, x2))

PADDING = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING

plt.xlim(x_axis_min, x_axis_max)
plt.ylim(y_axis_min, y_axis_max)
plt.rcParams["figure.figsize"] = (10, 10)
plt.show()