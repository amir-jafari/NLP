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
# -------------------------------WordNet -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# WordNet is a lexical database for the English language, which was created by Princeton, and is part of the NLTK corpus.

from nltk.corpus import wordnet

syns = wordnet.synsets("program")

print(syns[0].name())


print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())

synonyms = []
antonyms = []


for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))


w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))


#==================================================================================

from nltk.corpus import wordnet as wn

wn.synsets('motorcar')


wn.synset('car.n.01').lemma_names()
wn.synset('car.n.02').lemma_names()


wn.synset('car.n.01').definition()
wn.synset('car.n.01').examples()


#==================================================================================

wn.synsets('car')


for synset in wn.synsets('car'):
    print(synset.lemma_names())

wn.lemmas('car')