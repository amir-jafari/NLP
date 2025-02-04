from nltk import word_tokenize
from nltk import Text
import matplotlib.pyplot as plt

f = open('Corpus.txt')
raw = f.read()
f = open('Corpus.txt', 'r')
for line in f:
    print(line.strip())

words_token = word_tokenize(raw)
text = Text(words_token)
plt.figure(1,figsize=(4,2))
text.dispersion_plot(['corpus'])
plt.show()
