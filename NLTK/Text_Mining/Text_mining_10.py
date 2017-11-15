# %%%%%%%%%%%%% Natural Language Processing %%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 11 - 8 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Text Mining %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------N gram --------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# In the fields of computational linguistics and probability, an n-gram is a contiguous sequence of n items from a
# given sequence of text or speech. The items can be phonemes, syllables, letters, words or base pairs according
# to the application. The n-grams typically are collected from a text or speech corpus. When the items are words,
# n-grams may also be called shingles.
#
# An n-gram of size 1 is referred to as a "unigram"; size 2 is a "bigram" (or, less commonly, a "digram"); size 3
# is a "trigram". Larger sizes are sometimes referred to by the value of n in modern language, e.g., "four-gram",
# "five-gram", and so on.
# ----------------------------------------------------------------------------------------------------------------------
# Comparing and searching strings
# ----------------------------------------------------------------------------------------------------------------------
import ngram
import nltk
from nltk.book import *

ngram.NGram.compare('Ham','Spam', N=1)

G = ngram.NGram(['joe','joseph','jon','john','sally'])
print(G.search('jon'))
print(G.search('jon', threshold=0.3))
print(G.find('jose'))

# ----------------------------------------------------------------------------------------------------------------------
# Transforming items
# ----------------------------------------------------------------------------------------------------------------------
def lower(s):
	return s.lower()
G = ngram.NGram(key=lower)
print(G.key('AbC'))

print(G.pad('abc'))
list(G.split('abc'))
# ----------------------------------------------------------------------------------------------------------------------
# Set Operations
# ----------------------------------------------------------------------------------------------------------------------
G = ngram.NGram(['joe','joseph','jon','john','sally'])
G.update(['jonathan'])
print(sorted(list(G)))
print(G.discard('sally'))
print(sorted(list(G)))
G.difference_update(ngram.NGram(['joe']))
print(sorted(list(G)))
G.intersection_update(['james', 'joseph', 'joe', 'jon'])
print(sorted(list(G)))
G.symmetric_difference_update(ngram.NGram(['jimmy', 'jon']))
print(sorted(list(G)))

# ----------------------------------------------------------------------------------------------------------------------
# Bigram
# ----------------------------------------------------------------------------------------------------------------------

nltk.bigrams(['more', 'is', 'said', 'than', 'done'])

list(nltk.bigrams(['more', 'is', 'said', 'than', 'done']))

lb = list(nltk.bigrams(text1))
sent = ['This', 'sentence', 'will', 'give', 'us', 'some', 'bigrams', 'as', 'an', 'output', '.']
list(nltk.bigrams(sent))


list(nltk.bigrams('ham'))
list(nltk.ngrams(['ham','sam','bam','amir'],2))
