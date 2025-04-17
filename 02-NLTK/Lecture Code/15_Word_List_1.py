import nltk
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)
print(unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt')))
print( unusual_words(nltk.corpus.nps_chat.words()))

from nltk.corpus import stopwords
print(stopwords.words('english'))

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)
print(content_fraction(nltk.corpus.reuters.words()))