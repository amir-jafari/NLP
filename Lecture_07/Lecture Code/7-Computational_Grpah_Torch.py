import torch
from torch.autograd import Variable
x = Variable(torch.randn(3,4), requires_grad = True)
y = Variable(torch.randn(3,4), requires_grad = True)
z = Variable(torch.randn(3,4), requires_grad = True)
a = x * y
b = a + z
c = torch.sum(b)

c.backward()
print(x.grad.data)
print(y.grad.data)
print(z.grad.data)
# ----------------------------------------
from torch import nn
class model(nn.Module):
    def __init__(self, hidden_dim):
        super(model, self).__init__()
        self.linear1 = nn.Linear(1, hidden_dim)
        self.act1 = torch.sigmoid
        self.linear2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return self.linear2(self.act1(self.linear1(x)))
# ----------------------------------------
import torch
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
# ----------------------------------------
def train_loop (dataloader, model, loss, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = criterion(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()