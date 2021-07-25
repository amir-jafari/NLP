from nltk.corpus import inaugural
print(inaugural.fileids())
print([fileid[:4] for fileid in inaugural.fileids()])

import nltk
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))
cfd.plot()