import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
#  %%-----------------------------------------
p = np.linspace(start=-8, stop=8, num=1000)
#  %%-----------------------------------------
def softmax_f(n):
    return  np.exp(n)/sum(np.exp(n))

#  %%-----------------------------------------
a = np.array([0.1,0.2,0.7])
t_m = softmax_f(a)
t_s =softmax(a)
print(t_m)
print(t_s)
print()
#  %%-----------------------------------------

A= np.array([[0.1,0.2,0.7],
            [0.8,0.1,0.1],
            [0.1,0.7,0.2]])

print()
t_a0 =softmax(A,axis=0)
t_a1 =softmax(A,axis=1)
print(t_a0)
print()
print(t_a1)




