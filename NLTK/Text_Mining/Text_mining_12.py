# %%%%%%%%%%%%% Natural Language Processing %%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 11 - 8 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Text Mining %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
from collections import defaultdict
import nltk

def POS():
    return 'NOUN'
POS()


pos = defaultdict(POS)
pos['colorless'] = 'ADJ'
pos['amir']

# -------------------------------------------------------

list(pos.items())


def default_POS():
    return 'UNK'

alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v_100 = [word for (word, dummy) in vocab.most_common(1000)]

# -------------------------------------------------------

print(v_100[:20])

mapping = defaultdict(default_POS)
for v in v_100:
    mapping[v] = v
print(mapping)


alice2 = [mapping[v] for v in alice]
print(alice2[:100])

# -------------------------------------------------------


last_letters = defaultdict(list)

words = nltk.corpus.words.words('en')

for word in words:
    key = word[-2:]
    last_letters[key].append(word)

# Let's try out our function!
print(last_letters['ly'][:10])
print(last_letters['zy'][:10])
