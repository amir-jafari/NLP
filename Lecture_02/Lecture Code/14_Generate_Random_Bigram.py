import nltk
sent = ['In', 'the', 'beginning', 'God', 'created',
        'the', 'heaven', 'and', 'the', 'earth', '.']
print(list(nltk.bigrams(sent)))

def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
text1 = text[0:]
bigrams = nltk.bigrams(text)
print(list(nltk.bigrams(text1))[0:20])
cfd = nltk.ConditionalFreqDist(bigrams)
print(cfd['living'])
generate_model(cfd, 'living')