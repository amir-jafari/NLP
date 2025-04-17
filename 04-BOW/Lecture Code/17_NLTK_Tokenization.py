from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
sentence ='Thomas Jefferson began building Monticello at theage of 26.'
print(tokenizer.tokenize(sentence))

from nltk.tokenize import TreebankWordTokenizer
sentence = "Monticello wasn't designated as UNESCO World Heritage Site until 1987."
tokenizer = TreebankWordTokenizer()
print(tokenizer.tokenize(sentence))