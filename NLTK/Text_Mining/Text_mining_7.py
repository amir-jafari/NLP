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
# ------------------------------- Corpora and Lexical-------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import nltk
fields = nltk.corpus.gutenberg.fileids()
print(fields)


first_text = fields[0]
print(first_text)
print(len(first_text))
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------- gutenberg-----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from nltk.corpus import gutenberg
G_F = gutenberg.fileids()

dir(gutenberg)
# it has raw, words and sents as method

for field in G_F:
    num_chars = len(gutenberg.raw(field))
    num_words = len(gutenberg.words(field))
    num_sents = len(gutenberg.sents(field))
    num_vocab = len(set(w.lower() for w in gutenberg.words(field)))
    print('# Chars', num_chars,'# words', num_words, '# sentens', num_sents,'# vocabs', num_vocab, '-- name of fields',  field)

# ----------------------------------------------------------------------------------------------------------------------

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

Text1 = state_union.raw("2005-GWBush.txt")
Text2 = state_union.raw("2006-GWBush.txt")

ST = PunktSentenceTokenizer(Text1)

Tok = ST.tokenize(Text1)


for i in Tok:
    words = nltk.word_tokenize(i)
    tag = nltk.pos_tag(words)
    print(tag)


