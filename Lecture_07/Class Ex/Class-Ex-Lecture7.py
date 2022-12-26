# =================================================================
# Class_Ex1:
# Use glove word embeddings to train the MLP of the example
# % --------------------------------------------------------

# 1. Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/, unzip it and move glove.6B.50d.txt to the
# current working directory.

# 2. Define a function that takes as input the vocab dict from the example and returns an embedding dict with the token
# ids from vocab dict as keys and the 50-dim Tensors from the glove embeddings as values.

# 3. Define a function to return a Tensor that contains the tensors corresponding to the glove embeddings for the tokens
# in our vocabulary. The ones not found on the glove vocabulary are given tensors of 0s. This will happen more often
# than expected because our tokenizer is different from the one used for glove.

# 4. Replace the embedding weights of the model with the loop-up table returned by the function defined in 4. Check some
# of these vectors visually against the glove.6B.50d.txt file to make sure the correct embeddings are being used.

# 6. Add an option to freeze the embeddings so that they are not learnt. This will result in a poor performance because
# there are quite a few tokens which we don't have glove embeddings for (as mentioned in 4.), so we need to learn these.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')
































print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2:
# Use the following corpus
#
# corpus = ['king is a strong man',
#           'queen is a wise woman',
#           'boy is a young man',
#           'girl is a young woman',
#           'prince is a young king',
#           'princess is a young queen',
#           'man is strong',
#           'woman is pretty',
#           'prince is a boy will be king',
#           'princess is a girl will be queen']
# Train a  two layer neural network and show the the result of word2vec.
# Hint:
# 1- Remove the stop words
# 2- Use binary encoding for each word
# 3- Try a window size of 2 and 3
# 4- Make the embedding size of 2. Plot the each word and explain the results
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import numpy
# -------------------------------------------------------------------------------------














print(20*'-' + 'End Q2' + 20*'-')






