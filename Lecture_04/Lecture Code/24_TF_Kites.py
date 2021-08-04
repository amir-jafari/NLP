import nltk
from collections import Counter
from nltk.tokenize import TreebankWordTokenizer

with open('kite.txt','r') as f:
    kite_text = f.read()

tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(kite_text.lower())
token_counts = Counter(tokens)
print(token_counts)
print(20 *'-')
nltk.download('stopwords', quiet=True)
stopwords = nltk.corpus.stopwords.words('english')
tokens = [x for x in tokens if x not in stopwords]
kite_counts = Counter(tokens)
print(kite_counts)
