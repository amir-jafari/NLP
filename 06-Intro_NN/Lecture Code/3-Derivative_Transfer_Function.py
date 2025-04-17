import numpy as np
import matplotlib.pyplot as plt
#  %%-----------------------------------------
p = np.linspace(start=-8, stop=8, num=1000)
#  %%-----------------------------------------
def poslin(n):
    return np.maximum(0, n)

def purelin(n):
    return n

def logsig(n):
    return  1/(1 + np.exp(-n))

def softmax_f(n):
    return  np.exp(n)/sum(np.exp(n))

#  %%-----------------------------------------
t_poslin = poslin(p)
t_poslin_g = np.gradient(t_poslin)

plt.plot(p, t_poslin, ls='-')
plt.xlabel('p')
plt.ylabel('t')
plt.title('Transfer Function')
plt.show()

plt.plot(p, t_poslin_g, ls='-')
plt.xlabel('p')
plt.ylabel('t')
plt.title('Transfer Function Derivative')
plt.show()
#  %%-----------------------------------------
t_logsig = logsig(p)
t_logsig_g = np.gradient(t_logsig)

plt.plot(p, t_logsig, ls='-')
plt.xlabel('p')
plt.ylabel('t')
plt.title('Transfer Function')
plt.show()

plt.plot(p, t_logsig_g, ls='-')
plt.xlabel('p')
plt.ylabel('t')
plt.title('Transfer Function Derivative')
plt.show()


