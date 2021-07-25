import nltk
from nltk import word_tokenize

text = word_tokenize("And now for something completely different")
print(text)
tagged = nltk.pos_tag(text)
print(tagged)
print(nltk.help.upenn_tagset('RB'))


text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
print(text.similar('woman'))
print(text.similar('bought'))
print(text.similar('over'))
print(text.similar('the'))