import nltk
entries = nltk.corpus.cmudict.entries()

def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]
print( [w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']])
print([x[1] for x in entries if x[0]=='abbreviated'])

p3 = [(pron[0]+'-'+pron[2], word)
       for (word, pron) in entries
       if pron[0] == 'P' and len(pron) == 3]
cfd = nltk.ConditionalFreqDist(p3)
for template in sorted(cfd.conditions()):
     if len(cfd[template]) > 10:
        words = sorted(cfd[template])
        wordstring = ' '.join(words)
        print(template, wordstring[:70] + "...")
prondict = nltk.corpus.cmudict.dict()
prondict['blog'] = [['B', 'L', 'AA1', 'G']]
print(prondict['blog'])

