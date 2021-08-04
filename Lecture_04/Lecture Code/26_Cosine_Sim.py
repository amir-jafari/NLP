import numpy as np
from numpy.linalg import norm
def cosine_sim(a,b):
    return  np.dot(a, b)/(norm(a)*norm(b))
print(cosine_sim([1,1], [1,1]))
print(cosine_sim([1,1], [-1,1]))
print(cosine_sim([1,1], [-1,-1]))