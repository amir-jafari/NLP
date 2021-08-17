import torch

T1 = torch.ones(4,4)
print(T1[:,0])
print(T1[:,-1])
print(T1[...,-1])

T2 = torch.rand(4,1)
T3 = T1 @ T2 ; print(T3)
T4 = T1.matmul(T2); print(T4)

T5 = torch.randn(4,4)
T6 = T1 * T5; print(T6)
T7 = T1.mul(T5); print(T7)


