import torch
T = torch.randn(3,4)
print(T.dtype)
print(T.shape)
print(T.device)

if torch.cuda.is_available():
        tensor_gpu = T.to('cuda')
else:
    print('Tensors are not in the GPU.')