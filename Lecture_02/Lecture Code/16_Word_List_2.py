import nltk
names = nltk.corpus.names
print(names.fileids())
male_names = names.words('male.txt')
female_names = names.words('female.txt')
print([w for w in male_names if w in female_names])

cfd = nltk.ConditionalFreqDist(
         (fileid, name[-1])
         for fileid in names.fileids()
         for name in names.words(fileid))
cfd.plot()