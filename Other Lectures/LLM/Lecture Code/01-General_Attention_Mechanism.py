#%% --------------------------------------------------------------------------------------------------------------------
from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax

#%% --------------------------------------------------------------------------------------------------------------------
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

#%% --------------------------------------------------------------------------------------------------------------------
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

query_1 = word_1 @ W_Q
key_1 = word_1 @ W_K
value_1 = word_1 @ W_V

query_2 = word_2 @ W_Q
key_2 = word_2 @ W_K
value_2 = word_2 @ W_V

query_3 = word_3 @ W_Q
key_3 = word_3 @ W_K
value_3 = word_3 @ W_V

query_4 = word_4 @ W_Q
key_4 = word_4 @ W_K
value_4 = word_4 @ W_V

#%% --------------------------------------------------------------------------------------------------------------------
scores = array([dot(query_1, key_1), dot(query_1, key_2), dot(query_1, key_3), dot(query_1, key_4)])
print(scores)

#%% --------------------------------------------------------------------------------------------------------------------
weights = softmax(scores / key_1.shape[0] ** 0.5)
print(weights)

#%% --------------------------------------------------------------------------------------------------------------------
attention = (weights[0] * value_1) + (weights[1] * value_2) + (weights[2] * value_3) + (weights[3] * value_4)
print(attention)