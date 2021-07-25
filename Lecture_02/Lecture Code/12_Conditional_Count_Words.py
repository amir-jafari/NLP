from nltk.corpus import brown
import nltk

genre_word = [(genre, word)
           for genre in ['news', 'romance']
           for word in brown.words(categories=genre)]

print( genre_word[:4]); print(genre_word[-4:])
cfd = nltk.ConditionalFreqDist(genre_word)
print(cfd.conditions())

print(cfd['news'])
print(cfd['romance'])
cfd['romance'].most_common(20)