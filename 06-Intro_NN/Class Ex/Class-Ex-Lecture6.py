# =================================================================
# Class_Ex1:
# Lets consider the 2 following sentences
# Sentence 1: I  am excited about the perceptron network.
# Sentence 2: we will not test the classifier with real data.
# Design your bag of words set and create your input set.
# Choose your BOW words that suits perceptron network.
# Design your classes that Sent 1 has positive sentiment and sent 2 has a negative sentiment.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Use the same data in Example 1 but instead of hard-lim use log sigmoid as transfer function.
# modify your code inorder to classify negative and positive sentences correctly.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')





print(20 * '-' + 'End Q2' + 20 * '-')

# =================================================================
# Class_Ex2_1:

# For preprocessing, the text data is vectorized into feature vectors using a bag-of-words approach.
# Each sentence is converted into a vector where each element represents the frequency of a word from the vocabulary.
# This allows the textual data to be fed into the perceptron model.

# The training data consists of sample text sentences and corresponding sentiment labels (positive or negative).
# The text is vectorized and used to train the Perceptron model to associate words with positive/negative sentiment.

# For making predictions, new text input is vectorized using the same vocabulary. Then the Perceptron model makes a
# binary prediction on whether the new text has positive or negative sentiment.
# The output is based on whether the dot product of the input vector with the trained weight vectors is positive
# or negative.

# This provides a simple perceptron model for binary sentiment classification on textual data. The vectorization
# allows text to be converted into numerical features that the perceptron model can process. Overall,
# it demonstrates how a perceptron can be used for an NLP text classification task.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2_1' + 20 * '-')
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        print('TO DO')

    def predict(self, X):
        print('TO DO')
        return


# Sample training data
X_train = np.array([
    "I loved this movie, it was so much fun!",
    "The food at this restaurant is not good. Don't go there!",
    "The new iPhone looks amazing, can't wait to get my hands on it."
])
y_train = np.array([1, -1, 1])




print(20 * '-' + 'End Q2_1' + 20 * '-')

# =================================================================
# Class_Ex3:
# The following function is given
# F(x) = x1^2 + 2 x1 x2 + 2 x2^2 +x1
# use the steepest decent algorithm to find the minimum of the function.
# Plot the function in 3d and then plot the counter plot with the all the steps.
# use small value as a learning rate.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')





print(20 * '-' + 'End Q3' + 20 * '-')

# =================================================================
# Class_Ex4:
# Use the following corpus of data
# sent1 : 'This is a sentence one, and I want to all data here.',
# sent2 :  'Natural language processing has nice tools for text mining and text classification.
#           I need to work hard and try a lot of exercises.',
# sent3 :  'Ohhhhhh what',
# sent4 :  'I am not sure what I am doing here.',
# sent5 :  'Neural Network is a power method. It is a very flexible architecture'

# Train ADALINE network to find  a relationship between POS (just verbs and nouns) and the length of the sentences.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Read the dataset.csv file. This dataset is about the EmailSpam.
# Use a two layer network and to classify each email
# You are not allowed to use any NN packages.
# You can use previous NLP packages to read the data process it (NLTK, spaCY)
# Show the classification report and mse of training and testing.
# Try to improve your F1 score. Explain which methods you used.
# Hint. Clean the dataset, use all the preprocessing techniques that you learned.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')





print(20 * '-' + 'End Q5' + 20 * '-')
# =================================================================
# Class_Ex6:

# Follow the below instruction for writing the auto encoder code.

#The code implements a basic autoencoder model to learn word vector representations (word2vec style embeddings).
# It takes sentences of words as input and maps each word to an index in a vocabulary dictionary.

#The model has an encoder portion which converts word indexes into a low dimensional embedding via a learned weight
# matrix W1. This embedding is fed through another weight matrix W2 to a hidden layer.

#The decoder portion maps the hidden representation back to the original word index space via weight matrix W3.

#The model is trained to reconstruct the original word indexes from the hidden embedding by minimizing the
# reconstruction loss using backpropagation.

#After training, the weight matrix W1 contains the word embeddings that map words in the vocabulary to dense
# vector representations. These learned embeddings encode semantic meaning and can be used as features for
# downstream NLP tasks.


# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')





print(20 * '-' + 'End Q6' + 20 * '-')

# =================================================================
# Class_Ex7:
#
# The objective of this exercise to show the inner workings of Word2Vec in python using numpy.
# Do not be using any other libraries for that.
# We are not looking at efficient implementation, the purpose here is to understand the mechanism
# behind it. You can find the official paper here. https://arxiv.org/pdf/1301.3781.pdf
# The main component of your code should be the followings:
# Set your hyper-parameters
# Data Preparation (Read text file)
# Generate training data (indexing to an integer and the onehot encoding )
# Forward and backward steps of the autoencoder network
# Calculate the error
# look at error at by varying hidden dimensions and window size
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')



print(20 * '-' + 'End Q7' + 20 * '-')