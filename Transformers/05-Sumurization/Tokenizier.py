import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings(action='ignore')
# ----------------------------------------------------------------------------------------------------------------------
sample = open("text.txt", "r")
s = sample.read()
f = s.replace("\n", " ")

data = []
for i in sent_tokenize(f):
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)


model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=512, window=5, sg=1)
print(model2)
# ----------------------------------------------------------------------------------------------------------------------
def similarity(word1, word2):
    cosine = False
    try:
        a = model2.wv.get_vector(word1)
        cosine = True
    except KeyError:
        print(word1, ":[unk] key not found in dictionary")  # False implied

    try:
        b =model2.wv.get_vector(word2)
    except KeyError:
        cosine = False
        print(word2, ":[unk] key not found in dictionary")
    if (cosine == True):
        b = model2.wv.get_vector(word2)
        dot = np.dot(a, b)
        norma = np.linalg.norm(a)
        normb = np.linalg.norm(b)
        cos = dot / (norma * normb)

        aa = a.reshape(1, 512)
        ba = b.reshape(1, 512)
        # print("Word1",aa)
        # print("Word2",ba)
        cos_lib = cosine_similarity(aa, ba)
        # print(cos_lib,"word similarity")

    if (cosine == False): cos_lib = 0;
    return cos_lib
# ----------------------------------------------------------------------------------------------------------------------
word1 = "freedom";
word2 = "liberty"
print("Similarity", similarity(word1, word2), word1, word2)
# ----------------------------------------------------------------------------------------------------------------------
word1 = "corporations";
word2 = "rights"
print("Similarity", similarity(word1, word2), word1, word2)
# ----------------------------------------------------------------------------------------------------------------------
word1 = "etext";
word2 = "declaration"
print("Similarity", similarity(word1, word2), word1, word2)
# ----------------------------------------------------------------------------------------------------------------------
word1 = "justiciar";
word2 = "judgement"
print("Similarity", similarity(word1, word2), word1, word2)
# ----------------------------------------------------------------------------------------------------------------------
word1 = "judge";
word2 = "judgement"
print("Similarity", similarity(word1, word2), word1, word2)
# ----------------------------------------------------------------------------------------------------------------------
word1 = "justiciar";
word2 = "judge"
print("Similarity", similarity(word1, word2), word1, word2)
# ----------------------------------------------------------------------------------------------------------------------
word1 = "pay";
word2 = "debt"
print("Similarity", similarity(word1, word2), word1, word2)

