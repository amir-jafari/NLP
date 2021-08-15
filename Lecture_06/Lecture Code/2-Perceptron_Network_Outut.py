import numpy as np
import matplotlib.pyplot as plt
#  %%-----------------------------------------
p = np.linspace(start=-8, stop=8, num=1000)
#  %%-----------------------------------------
def hardlim(n):
    return np.where(n >= 0.0, 1, 0)
t_hardlim = hardlim(p)

plt.plot(p, t_hardlim, ls='-')
plt.xlabel('p')
plt.ylabel('t')
plt.title('Transfer Function')
plt.show()
#  %%-----------------------------------------
p = np.array([[1,1],
              [1,2],
              [2,1],
              [-1,-1],
              [-1,-2],
              [-2,-1]])

W = np.array([[1 ,1],
              [-1,-1]])
b = np.array([[1],
              [-1]])

n = np.array([W @ x.reshape(-1,1) + b for x in p])
a= hardlim(n)

plt.figure()
for i in range(len(a)):
    if a[:,:,-1][i][0] >=1:
        plt.scatter(p[i,0],p[i,1], color='red')
    else:
        plt.scatter(p[i,0],p[i,1], color='green')
plt.show()

