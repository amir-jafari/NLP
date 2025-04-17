from nltk.book import *
import nltk

print(text1.vocab())
print(type(text1))
print(len(text1))

from nltk.corpus import gutenberg
print(gutenberg.fileids())
print(nltk.corpus.gutenberg.fileids())
hamlet = gutenberg.words('shakespeare-hamlet.txt')

from nltk.corpus import inaugural
print(inaugural.fileids())
print(nltk.corpus.inaugural.fileids())
from nltk.text import Text
former_president = Text(inaugural.words(inaugural.fileids()[-1]))
print(' '.join(former_president.tokens[0:1000]))

