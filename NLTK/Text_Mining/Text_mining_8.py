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
# -------------------------------Stop Words-----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

print(stop_words)

word_tokens = word_tokenize(example_sent)

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)


stopwords.words('english')
# ----------------------------------------------------------------------------------
def content_fraction(text):
     stopwords = nltk.corpus.stopwords.words('english')
     content = [w for w in text if w.lower() not in stopwords]
     return len(content) / len(text)

content_fraction(nltk.corpus.gutenberg.words('austen-emma.txt'))


def remove_stopwords(text):
    text = set(w.lower() for w in text if w.isalpha())
    stopword_vocab = nltk.corpus.stopwords.words('english')
    no_stop = text.difference(stopword_vocab)
    return no_stop

emma_unique_nostops = remove_stopwords(nltk.corpus.gutenberg.words('austen-emma.txt'))
