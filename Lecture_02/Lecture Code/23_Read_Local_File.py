from nltk import word_tokenize
from nltk import Text

f = open('Corpus.txt')
raw = f.read()
f = open('Corpus.txt', 'r')
for line in f:
    print(line.strip())

words_token = word_tokenize(raw)
text = Text(words_token)
text.dispersion_plot(['corpus'])
