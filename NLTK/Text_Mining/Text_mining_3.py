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

