# %%%%%%%%%%%%% Natural Language Processing %%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 11 - 8 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Text Mining %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# ---------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------Tokenizing-----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# tokenizing - word tokenizers ...... sentence tokenizers

# Corpus - Body of text
# Lexicon - Words and their meanings.
# Token - Each "entity" that is a part of whatever was split up based on rules.
# ----------------------------------------------------------------------------------------------------------------------
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

EXAMPLE_TEXT = "Hello Every one, How is it going? Python is awesome and we are learning it. This is Python script for this Course."


Word_Tok = word_tokenize(EXAMPLE_TEXT)
for i in Word_Tok:
    print(i)

Sen_Tok = sent_tokenize(EXAMPLE_TEXT)

for i in Sen_Tok:
    print(i)

print("All words in a list = ", Word_Tok)
print("All sentences in a list = ", Sen_Tok)

# Do the dir(Sen_Tok) and same thing on the Word_Tok the figure out these objects are what data frames then
# try to use some methods on it. Since these are lis the list method will work perfectly fine.
# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------Stemming------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

from nltk.stem.porter import *
PS = PorterStemmer()

EXAMPLE_TEXT = ['get', 'getting', 'got','gotten']

for w in EXAMPLE_TEXT:
    print(PS.stem(w))

porter = nltk.PorterStemmer()


def unusual_words(text):
    text_vocab = set(porter.stem(w).lower() for w in text if w.isalpha())
    english_vocab = set(porter.stem(w).lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)


unusual_words(nltk.corpus.gutenberg.words('austen-emma.txt'))

from nltk.stem.porter import *

stemmer = PorterStemmer()
plurals = ['caresses', 'flies', 'dies', 'mules', 'denied',
           'died', 'agreed', 'owned', 'humbled', 'sized',
           'meeting', 'stating', 'siezing', 'itemization',
           'sensational', 'traditional', 'reference', 'colonizer',
           'plotted']

singles = [stemmer.stem(plural) for plural in plurals]

# amir
from nltk.stem.porter import *

PS = PorterStemmer()

EXAMPLE_TEXT = ['pyhton', 'pythonly', 'pythoned', 'pythoning']

for w in EXAMPLE_TEXT:
    print(PS.stem(w))
# ----------------------------------------------------------------------------------------------------------------------

from nltk.stem.snowball import SnowballStemmer
print(" ".join(SnowballStemmer.languages))
SNB = stemmer = SnowballStemmer("english")

for w in EXAMPLE_TEXT:
    print(SNB.stem(w))
# ----------------------------------------------------------------------------------------------------------------------

from nltk.stem.lancaster import LancasterStemmer

LS = LancasterStemmer()

for w in EXAMPLE_TEXT:
    print(LS.stem(w))
# ----------------------------------------------------------------------------------------------------------------------

from nltk.stem import WordNetLemmatizer
WL = WordNetLemmatizer()

EXAMPLE_TEXT = ['gets', 'churches', 'breaked','brokken']

for w in EXAMPLE_TEXT:
    print(WL.lemmatize(w))

# ----------------------------------------------------------------------------------------------------------------------
# Lemmatizing
# ----------------------------------------------------------------------------------------------------------------------
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run", 'v'))




