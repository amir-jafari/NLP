import torch

t = torch.rand(2, 1, 2, 1); print(t)
r = torch.squeeze(t); print(r)
r = torch.squeeze(t, 1); print(r)


x = torch.rand([1, 2, 3]); print(x)
r = torch.unsqueeze(x, 0); print(r)
r = torch.unsqueeze(x, 1); print(r)

v = torch.arange(9).reshape(3,3)
# flatten a Tensor and return elements with given indexes
r = torch.take(v, torch.LongTensor([0, 4, 2]))
r = torch.transpose(v, 0, 1); print(r)

mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 4)
r = torch.mm(mat1, mat2)

v1 = torch.ones(3)
r = torch.diag(v1)