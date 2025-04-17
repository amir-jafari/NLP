from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
sentence = "The faster Harry got to the store, the faster Harry " \
           "the faster, would get home."
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
print(tokens)

bag_of_words = Counter(tokens)
print(bag_of_words)

