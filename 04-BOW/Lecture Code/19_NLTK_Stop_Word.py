import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)
print(len(stop_words))

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
print(len(sklearn_stop_words))
print(len(set(stop_words).union(sklearn_stop_words)))
print(len(set(stop_words).intersection(sklearn_stop_words)))