import nltk
from nltk import word_tokenize

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
 is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)

wnl = nltk.WordNetLemmatizer()
print([wnl.lemmatize(t) for t in tokens])