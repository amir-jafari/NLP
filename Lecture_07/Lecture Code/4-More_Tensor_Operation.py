import torch

x = torch.Tensor(2, 3) ; print(x)
y = torch.rand(2, 3)
z2 = torch.add(x, y); print(z2)
print(torch.is_tensor(z2))
z1 = torch.Tensor(2, 3)
torch.add(x, y, out=z1)

print(x.size())
print(torch.numel(x))

k = x.view(6); print(k)
l = x.view(-1, 2); print(l)

x1 = torch.randn(5, 3).type(torch.FloatTensor); print(x1)
x2 = torch.randn(5, 3).type(torch.LongTensor); print(x2)

v = torch.arange(9); print(v)
r1 = torch.cat((x, x, x), 0); print(r1)
r2 = torch.stack((v, v)); print(r2)
r3 = torch.chunk(v, 3); print(r3)