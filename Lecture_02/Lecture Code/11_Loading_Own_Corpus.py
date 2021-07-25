from nltk.corpus import PlaintextCorpusReader
import os

corpus_root = os.getcwd()
wordlists = PlaintextCorpusReader(corpus_root, 'Corpus.txt')

print(wordlists.fileids())
print(wordlists.words('Corpus.txt'))