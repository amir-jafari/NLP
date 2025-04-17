import nltk

text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = nltk.sent_tokenize(text)
print(sents[0])


sentences = text.split('. ')
words_in_sentences = [sentence.split(' ') for sentence in sentences]
print(sentences[0])