# %%%%%%%%%%%%% Natural Language Processing %%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 11 - 8 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Text Mining %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#

#==================================================================================
#%%
#  Generating bigrams

# Generate a bigram from a sentence
import nltk

sent = ['This', 'sentence', 'will', 'give', 'us', 'some', 'bigrams', 'as', 'an', 'output', '.']
list(nltk.bigrams(sent))


list(nltk.bigrams('ham'))
list(nltk.ngrams(['ham','sam','bam','amir'],2))
#==================================================================================
#%%
# Generating bigrams

# What if we want to find the most frequently occurring words for a 
# particular word?

def generate_model(cfdist, word, num = 15):   # First, define inputs
    for i in range(num):                      # For i in the range of num
        print(word)
        word = cfdist[word].max()  
           
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
bigrams = nltk.bigrams(emma)
cfd = nltk.ConditionalFreqDist(bigrams) 

cfd['lovely']
generate_model(cfd, 'lovely')

#==================================================================================
#%%
# Getting plural words

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


#==================================================================================
#%%
#
####################################################################
####                                                            ####
#### Exercise 1: Adjusting the function to include more plurals ####
####                                                            ####
####################################################################

'''
What is we want to turn 'goose' into a plural? Add another parameter
to the plural function with the 'elif...return' syntax to turn
'goose' into 'geese' and then test it out. You can copy and paste
the function above to adjust it
'''

#==================================================================================
#%%

my_text = ["Here", "are", "some", "words", "that", "are", "in", "a", "list"]
vocab = sorted(set(my_text))
word_freq = nltk.FreqDist(my_text)

vocab
word_freq



#==================================================================================
#%%
# 

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

EXAMPLE_TEXT = ['pyhton', 'pythonly', 'pythoned','pythoning']

for w in EXAMPLE_TEXT:
    print(PS.stem(w))
 
#==================================================================================
# 


from nltk.corpus import stopwords
stopwords.words('english')

def content_fraction(text):
     stopwords = nltk.corpus.stopwords.words('english')
     content = [w for w in text if w.lower() not in stopwords]
     return len(content) / len(text)

content_fraction(nltk.corpus.gutenberg.words('austen-emma.txt'))

#==================================================================================
#
# 

def remove_stopwords(text):
    text = set(w.lower() for w in text if w.isalpha())
    stopword_vocab = nltk.corpus.stopwords.words('english')
    no_stop = text.difference(stopword_vocab)
    return no_stop

emma_unique_nostops = remove_stopwords(nltk.corpus.gutenberg.words('austen-emma.txt'))

#==================================================================================
#%%
#
##################################################
####                                          ####
#### Exercise 2: Removing stopwords from text ####
####                                          ####
##################################################

'''
Rerun the two functions we created above with the Reuters corpus:
nltk.corpus.reuters.words()

What did you notice about the stopwords from this corpus? Why would
this number occur?

'''




'''
BONUS: Create a function that only pulls out the stopwords from the
Reuters text - name this function 'identify_stopwords'. 
Hint: you may want to think about the syntax 
list(set(text1) & set(text2)) to find overlap. Use the remove_stopwords function
as a reference.

Test out your function on nltk.corpus.reuters.words('test/14829')
to see what stopwords are in the article
'''    


#==================================================================================
#%%
# Identifying female and male names

# It's important to be able to cross-reference dictionaries - let's
# use the Names Corpus that contains 8,000 names categorized by gender.
# Let's find the names that appear in both the male and female names
# lists.
names = nltk.corpus.names
names.fileids()

male_names = names.words('male.txt')
female_names = names.words('female.txt')

both_names = [w for w in male_names if w in female_names]
both_names

#==================================================================================
#%%
# Identifying female and male names

# Usually, names that end in vowels correspond to female names, while
# male names don't. Let's map that out and see if it holds true.
cfd = nltk.ConditionalFreqDist(
           (fileid, name[-1])
           for fileid in names.fileids()
           for name in names.words(fileid))

cfd.plot()

#==================================================================================
#%%
# Pronouncing dictionary

# There is a dictionary that contains additional information about 
# the pronunciation of words - let's look at some of the entries.

entries = nltk.corpus.cmudict.entries()
len(entries)

for entry in entries[42371:42379]:
    print(entry) 

#==================================================================================
#%%
# Pronouncing dictionary
    
# Let's get more specific and use 2 variable names - here, we're saying
# if there are 3 phonetic codes (pron) and the first pron is 'P', and the
# third one is 'T', print the word and the middle phonetic code     
for word, pron in entries:
    if len(pron) == 3:
        ph1, ph2, ph3 = pron
        if ph1 == 'P' and ph3 == 'T':
            print(word, ph2, '\n')

#==================================================================================
# %%
#  Pronouncing dictionary

# Let's pull out all the words that end in an 'x' sound.
syllable = ['N', 'IH0', 'K', 'S']
[word for word, pron in entries if pron[-4:] == syllable]
#  pron[-4:] pulls the last four syllable pronunciations of each word.
#  So we are comparing the last four syllables of each word to the 
#   above list of syllables ("n i k s")

# What about words that end in an 'M' sound, but end in the letter 'n'.
[w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']

#==================================================================================
#%%
#
##################################################
####                                          ####
####     Exercise 3: Finding rhymes           ####
####                                          ####
##################################################

# First find the pronunciation for "green" in the entries list
[(word, pron) for word, pron in entries if word=="green"]

green = ['G', 'R', 'IY1', 'N']

# Now find all words that rhyme with "green"
temp = ['A', 'B', 'C', 'D']
temp[-2:]
[word for word, pron in entries if pron[-3:] == ['R', 'IY1', 'N']]

#==================================================================================

from nltk.corpus import wordnet as wn

wn.synsets('motorcar') 

# The '01' at the end is the first set of definitions, 
# but there are other definitions as well.
wn.synset('car.n.01').lemma_names()
wn.synset('car.n.02').lemma_names()

# You can look up definitions and examples of the synset.
wn.synset('car.n.01').definition()
wn.synset('car.n.01').examples()


#==================================================================================
#%%
#  WordNet

# How many synsets is 'car' in?
wn.synsets('car')

# What are the synonyms for each synset?
for synset in wn.synsets('car'):
    print(synset.lemma_names())

# Which synsets have the lemma 'car'?
wn.lemmas('car')


#==================================================================================
#%%
#
##################################################
####                                          ####
####        Exercise 4: Using WordNet         ####
####                                          ####
##################################################

'''
Find all the synsets that contain 'book', how many are there?
What are the difference synsets for subset 01, 07, 05?
What are the synonyms for each synset?
What are some examples from synset 09?
'''



#==================================================================================

from urllib2 import urlopen

url = "http://www.gutenberg.org/files/2554/2554-0.txt"

# Then, open the URL and read the text, 
# specifying that it is UTF-8.
response = urlopen(url)
raw = response.read().decode('utf8')

type(raw) # What is the structure? 
len(raw)  # What is the length?  1176965
raw[:75]  # The first 75 characters?  

#==================================================================================
#%%
# Accessing text from the web

# First, import word_tokenize from nltk
from nltk import word_tokenize

# Next, tokenize the raw text we pulled from the URL
tokens = word_tokenize(raw)

# What does the tokenized text look like?
type(tokens)
len(tokens)
tokens[:10]

#==================================================================================
#%%
# Tokenizing text

# Transform the tokenized text from a 'list' to 'text'
text = nltk.Text(tokens)

# We can confirm that it's text format, and pull particular words out
type(text)
text[1024:1062]

# When we look at collocations, we can see word 
# pairs that occur frequently together
text.collocations()


#==================================================================================
#%%
#  Tokenizing text

# What if we only wanted to include the content, without any other
# headers? 

# First, we'll find the location for the beginning of the text.
raw.find("PART I")
# 5336 

# Then, we'll find the location for the end of the text.
raw.find("He had successfully avoided meeting his landlady on the staircase.")
# 5543

# We'll subset *only the characters **between** those locations*
raw_sub = raw[5336:5543]

print(raw_sub)
# Now, when we look for "PART I", it's at the zero index location
raw_sub.find("PART I")

# How many characters did we remove?
len(raw) - len(raw_sub)

#==================================================================================
#%%
# Tokenizing text

# Don't forget to tokenize the new text, and check the new length
# and the first 10 words.
tokens_sub = word_tokenize(raw_sub)

len(tokens_sub)
tokens_sub[:10]

# How many words did we remove?
len(tokens) - len(tokens_sub)

#==================================================================================
#%%
# Working with HTML

from bs4 import BeautifulSoup
from urllib2 import urlopen

url = "http://google.com"

raw_html = urlopen(url).read()
raw_html[:250]  # look at first 250 characters of the raw response

# Let's clean out the HTML
# use beautiful soup to parse the response into something useful
parsed_html = BeautifulSoup(raw_html, 'html.parser') 
body = parsed_html.body # the only element we probably care about is the body

#==================================================================================
#%%
# Working with HTML

# Beautiful Soup lets you extract elements by their tag. For example 'h1', 
# 'a', 'div', etc.
body('h1')  # returns a list of all the first level header elements in the body
[h1.get_text() for h1 in body('h1')]
# use get_text() on each of them to get the value inside the html tags

# we can also remove any <script> tags which contain code, not content
scripts = body('script') # view the script tags
scripts[2] # some type of javascript function, we're not interested in this
_ = [s.extract() for s in body('script')] # remove the script tags from body

body_text = body.get_text()

body_tokens = nltk.word_tokenize(body_text)

# When we check, we should see that the HTML is removed!
body_tokens[0:100]

#==================================================================================
#%%
# Working with HTML
from nltk import word_tokenize

# Let's extract just the <p> tags for paragraphs.
#  This most likely contains the article's content
# (though some websites don't always conform to this!)
pars = [p.get_text() for p in body('p')]
par_tokens = word_tokenize('\n'.join(pars))

# Then, we can reformat it as text, so we can perform functions on it.
pars_nltk = nltk.Text(par_tokens)
pars_nltk.concordance('work')

pars_nltk[:10]
pars_nltk[-20:]

#==================================================================================
#%%
#
##################################################
####                                          ####
####     Exercise 5: Working with HTML        ####
####                                          ####
##################################################

"""
Repeat the above process for a website of your choosing! 

"""

#==================================================================================
#%%
# Reading local files

# In order to read local files, you can use the open() syntax
f = open('document.txt')

# Next, read it in and then view it
raw = f.read()
raw

# Read in one line at a time.
f = open('document.txt', 'r')

# Use a simple for loop to print each line
for line in f:
    print(line.strip())
    
# Here we use the strip() method to remove the newline character at the end 
# of the input line.

'''
'r'       open for reading (default)
'w'       open for writing, truncating the file first
'x'       create a new file and open it for writing
'a'       open for writing, appending to the end of the file if it exists
'b'       binary mode
't'       text mode (default)
'+'       open a disk file for updating (reading and writing)
'U'       universal newline mode (deprecated, doesn't exist anymore)
'''

#==================================================================================
#%%
# Regular expressions

'''
To use regular expressions in Python, we need to import the re library 
using import re. We also need a list of words to search; we'll use the 
Words Corpus again. We will preprocess it to remove any proper names.
'''

import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

#==================================================================================
#%%
# Regular expressions

# Finding words that end in 'ed' - the first argument in search is
# the pattern we want to find, the second is the text we want to search.
[w for w in wordlist if re.search('ed$', w)]

# Using wildcards to find words - what words have 
# a 'j' as a third letter, and a 't' as a sixth letter?
[w for w in wordlist if re.search('^..j..t..$', w)]

# Finding multiple word combinations of the letters in each place
[w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]

#==================================================================================
#%%
# Regular expressions

# Let's look at the chat corpus to find multiple spellings of the same
# words.
chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))


[w for w in chat_words if re.search('^m+i+n+e+$', w)]

# Finding multiple word combinations for 'ha'.
#  -- what happens if you replace [] with () ?
[w for w in chat_words if re.search('^[ha]+$', w)]

#==================================================================================
#%%
#
##################################################
####                                          ####
####     Exercise 6: Regex                    ####
####                                          ####
##################################################

"""
Find all words that end in 'f'
"""



"""
Find all words that end in 'self'
"""



"""
Find all three letter words that contain either "s" or "a" (or both)
"""



#==================================================================================
#%%
# Finding word stems

# When text mining, you have to stem your words before analyzing it - 
# for example, 'table' and 'tables' would be read as two different words.
# Here is a way to strip out different suffixes from words.
def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem

stem('walking')
stem('walked')
stem('stated') # (this doesn't work very well)
stem('running')
#==================================================================================
#%%
# Finding word stems

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
 is no basis for a system of government.  Supreme executive power derives from
 a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)

[stem(t) for t in tokens]

#==================================================================================
#%%
# Searching tokenized text

# Let's import the gutenberg and chat texts.
from nltk.corpus import gutenberg, nps_chat

# Set 'moby' equal to Moby Dick text.
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))

# This creates a special nltk "Text" object, which will have
#  its own set of useful methods we can use
type(moby)


# Here, we're finding the words that appear between 'a' and 'man' - the
# <.*> matches a single token that will be returned.
moby.findall(r"<a> (<.*>) <man>")

# for example, we see "a nervous man", "a good man", "a younger man", etc.
# Note, this findall method is different than the standard one built into 
# Python's regular expression library. It is extended for nltk to be able
# to find words (tokens) easily, using the <...> notation
#==================================================================================
#%%
# Searching tokenized text

# Set 'chat' equal to the chat text.
chat = nltk.Text(nps_chat.words())

# Now, we'll find two words that appear before 'bro'
chat.findall(r"<.*> <.*> <bro>")

# Let's look for sequences of three words that all start with "l"
chat.findall(r"<l.*>{3,}")

#==================================================================================
#%%
#Normalizing text

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
 is no basis for a system of government.  Supreme executive power derives from
 a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)

# Let's use the off-the-shelf stemmers to stem the above paragraph.
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
# these libraries work much better than our simple 
# solution. Porter is considered the industry standard for stemming
# Lancaster is a similar alternative 

#==================================================================================
#%%
# Normalizing text

# Let's try stemming a few words with the Porter stemmer.
porter.stem('running') 
porter.stem('stated')

[porter.stem(t) for t in tokens]

[lancaster.stem(t) for t in tokens]

#==================================================================================
#%%
#  Lemmatization

# The WordNet lemmatizer is a good choice if you want to compile the 
# vocabulary of some texts and want a list of valid lemmas (or lexicon words
# that are found in the dictionary).
wnl = nltk.WordNetLemmatizer()
wnl.lemmatize('chairs')
wnl.lemmatize('said', pos='v')
wnl.lemmatize('are', pos='v')

# If we want to lemmatize a list of words, 
# this is not possible without knowing their part of speech,
# which we will learn how to do later.

#==================================================================================
#%%
# Lemmatization

# The WordNet lemmatizer is a good choice if you want to compile the 
# vocabulary of some texts and want a list of valid lemmas (or lexicon words
# that are found in the dictionary).
[wnl.lemmatize(t) for t in tokens]

#==================================================================================
#%%
#  Regular expressions for tokenization

raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
 though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
 well without--Maybe it's always pepper that makes people hot-tempered,'"""

re.split(r' ', raw)
re.split(r'[ \t\n]+', raw)

#==================================================================================

#  Regular expressions for tokenization

# We can use \W to split the input on anything other than a word character.
re.split(r'\W+', raw)


re.findall(r'\w+|\S\w*', raw)

#==================================================================================
#%%
# NLTK's RegEx tokenizer

# We can use NTLK's tokenizer and specify the text and 
# a pattern
text = 'That U.S.A. poster-print costs $12.40...'
pattern = r'\w+|\S\w*'

# Then, we can use regexp_tokenize on it to tokenize the text.
nltk.regexp_tokenize(text, pattern)



#==================================================================================
#%%
# Formatting lists to strings

# Here is our sentence.
sentence = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']

# We can join the words with a space.
' '.join(sentence)

# Here, we're joining the words with a semi-colon.
';'.join(sentence)

# Here, we're just joining the words, no spaces or punctuation marks.
''.join(sentence)


#%%
#
##################################################
####                                          ####
####     Exercise 7: Lemmatization            ####
####                                          ####
##################################################

# This exercise uses the following string of verbs
verbs = """ran
said
being
were
was
had
lost
walked
surprised
broke"""

# Take the following string of conjugated verbs, and break it into a list



# Make an empty list called cleaned_verbs


# Now loop through the list and print the lemma of each.
#  Also append the lemmatized verb to your cleaned_verbs list


    
# Print the cleaned_verbs list by joining each member together, separated
# with the vertical pipe character

    
# Bonus: how would you produce the following output?
# ran => run
# said => say
# etc...



