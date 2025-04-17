from nltk.corpus import brown
import nltk
print(brown.categories())
print(brown.words(categories='news'))
print( brown.words(fileids=['cg22']))
print(brown.sents(categories=['news', 'editorial', 'reviews']))
from nltk.corpus import brown
news_text = brown.words(categories='news')
fdist = nltk.FreqDist(w.lower() for w in news_text)
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m + ':', fdist[m], end=' ')

cfd = nltk.ConditionalFreqDist((genre, word)
            for genre in brown.categories()
            for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
print(); print(cfd.tabulate(conditions=genres, samples=modals))