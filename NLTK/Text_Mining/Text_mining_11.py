# %%%%%%%%%%%%% Natural Language Processing %%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 11 - 8 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Text Mining %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#

import nltk
from nltk.book import *
#==================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# Getting plural words
# ----------------------------------------------------------------------------------------------------------------------
def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'
 	
plural('fairy')
plural('woman')

# ----------------------------------------------------------------------------------------------------------------------
# Pronunciations
# ----------------------------------------------------------------------------------------------------------------------

entries = nltk.corpus.cmudict.entries()
len(entries)

for entry in entries[42371:42379]:
    print(entry) 


for word, pron in entries:
    if len(pron) == 3:
        ph1, ph2, ph3 = pron
        if ph1 == 'P' and ph3 == 'T':
            print(word, ph2, '\n')


syllable = ['N', 'IH0', 'K', 'S']
[word for word, pron in entries if pron[-4:] == syllable]
[w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']


# ----------------------------------------------------------------------------------------------------------------------
# Regular expressions
# ----------------------------------------------------------------------------------------------------------------------

'''
To use regular expressions in Python, we need to import the re library 
using import re. We also need a list of words to search; we'll use the 
Words Corpus again. We will preprocess it to remove any proper names.
'''

import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]


# Finding words that end in 'ed' - the first argument in search is
# the pattern we want to find, the second is the text we want to search.
[w for w in wordlist if re.search('ed$', w)]

# Using wildcards to find words - what words have 
# a 'j' as a third letter, and a 't' as a sixth letter?
[w for w in wordlist if re.search('^..j..t..$', w)]

# Finding multiple word combinations of the letters in each place
[w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]



chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
[w for w in chat_words if re.search('^m+i+n+e+$', w)]
[w for w in chat_words if re.search('^[ha]+$', w)]


# ----------------------------------------------------------------------------------------------------------------------
# Searching tokenized text
# ----------------------------------------------------------------------------------------------------------------------

# Set 'chat' equal to the chat text.
chat = nltk.Text(nps_chat.words())

# Now, we'll find two words that appear before 'bro'
chat.findall(r"<.*> <.*> <bro>")

# Let's look for sequences of three words that all start with "l"
chat.findall(r"<l.*>{3,}")



raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
 though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
 well without--Maybe it's always pepper that makes people hot-tempered,'"""

re.split(r' ', raw)
re.split(r'[ \t\n]+', raw)
re.split(r'\W+', raw)
re.findall(r'\w+|\S\w*', raw)


# NLTK's RegEx tokenizer
# We can use NTLK's tokenizer and specify the text and
# a pattern
text = 'That U.S.A. poster-print costs $12.40...'
pattern = r'\w+|\S\w*'

# Then, we can use regexp_tokenize on it to tokenize the text.
nltk.regexp_tokenize(text, pattern)










