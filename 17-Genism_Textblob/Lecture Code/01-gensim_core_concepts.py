#%% --------------------------------------------------------------------------------------------------------------------
import pprint
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities
#%% --------------------------------------------------------------------------------------------------------------------
# Document
print('\n')
print("=============================================Document in Gensim================================================")
document = "Human machine interface for lab abc computer applications"
print(document)
#%% --------------------------------------------------------------------------------------------------------------------
# Corpus
print('\n')
print("==============================================Corpus in Gensim=================================================")
text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]
stoplist = set('for a of the and to in'.split(' '))
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
#%% --------------------------------------------------------------------------------------------------------------------
# Vector
print('\n')
print("==============================================Vector in Gensim=================================================")
pprint.pprint(dictionary.token2id)
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)
#%% --------------------------------------------------------------------------------------------------------------------
# Model
print('\n')
print("==============================================Model in Gensim==================================================")
tfidf = models.TfidfModel(bow_corpus)
words = "system minors".lower().split()
print(tfidf[dictionary.doc2bow(words)])
index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)
query_document = 'system engineering'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
print(list(enumerate(sims)))
for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)