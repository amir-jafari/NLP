from nltk import word_tokenize
import nltk
from nltk import Text

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
 is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
print([porter.stem(t) for t in tokens])
print( [lancaster.stem(t) for t in tokens])

porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = Text(grail)
text.concordance('lie')