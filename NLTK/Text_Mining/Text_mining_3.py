# %%%%%%%%%%%%% Natural Language Processing %%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 11 - 8 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Text Mining %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# ---------------------------------------------------------------------------------------------------------------------

# Import nltk book

import nltk

from nltk.book import *
# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Dispersion plot-----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
text4.dispersion_plot(["China", "unity", "freedom", "USA", "America"])
text4.dispersion_plot(["China", "unity", "freedom", "us", "America"])
text4.dispersion_plot(["hate", "unity", "love", "us", "crime"])
text4.dispersion_plot(["tax", "immigration", "health", "care"])

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Frequency distribution----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

Freq_Dist = FreqDist(text1)
print(Freq_Dist)

Freq_Dist.most_common(50)
Freq_Dist['his']


Freq_Dist.plot(50, cumulative = False)
Freq_Dist.plot(50, cumulative = True)


Freq_Dist.hapaxes()
Once_happend= Freq_Dist.hapaxes()
text4.count('men') / float(len(text6) * 100)

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Finding words------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

Value_set = set(text1)

long_words = [words for words in Value_set if len(words) > 17]
sorted(long_words)
# ---------------------------
s1 = []
for i,v in enumerate(Value_set):
    if len(v) > 17:
        s1.append(v)
sorted(s1)

T = sorted(s1)==sorted(long_words)
print(T)
# ---------------------------
s2 = []

for i,v in enumerate(Value_set):
    if len(v) <= 1:
        s2.append(i)
V1 = list(Value_set)
s3 = []
for i in range(len(s2)):
    V1[s2[i]] = []
# ---------------------------
F_D1 = FreqDist(text2)
sorted(words1 for words1 in set(text2) if len(words1) > 7 and F_D1[words1] > 7)
# ---------------------------
my_text = ["Here", "are", "some", "words", "that", "are", "in", "a", "list"]
vocab = sorted(set(my_text))
word_freq = nltk.FreqDist(my_text)

print(vocab)
print(word_freq)

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Counting and Frequency----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

LC = [len(words2) for words2 in text1]
print(LC)



F_D2 = FreqDist(len(words2) for words2 in text1)
print(F_D2)
F_D2.plot(6)

# ---------------------------

F_D2.most_common()
F_D2.max()
F_D2[3]
F_D2.freq(3)

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Investigating Words------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
LC2 = [words3 for words3 in set(text1) if words3.endswith('ness')]
print(LC2)
print(sorted(LC2))
LC3 = [words4 for words4 in set(text2) if words4.endswith('ness')]
print(LC3)
print(sorted(LC3))
LC4 = [words5 for words5 in set(text3) if words5.endswith('ness')]
print(LC4)
print(sorted(LC4))

# ---------------------------

print(sorted(term for term in set(text4) if 'nt' in term))

print(sorted(item for item in set(text6) if item.istitle()))

print(sorted(item for item in set(sent7) if item.isdigit()))

print(sorted(w for w in set(text7) if '-' in w and 'index' in w))

print(sorted(wd for wd in set(text3) if wd.istitle() and len(wd) > 10))

print(sorted(w for w in set(sent7) if not w.islower()))

print(sorted(t for t in set(text2) if 'cie' in t or 'cei' in t))

# ---------------------------
ls = ['amir', '1', 'Martin', '2016', 'Hagan', ',']
a1 =[word.lower() for word in ls if word.isalpha()]
print(a1)
a2 = set(word.lower() for word in ls if word.isalpha())
print(len(a2))

for word in ls:
    if word.endswith('n'):
        print(word)

# ---------------------------

for word6 in ls:
    if word6.islower():
        print(word6, 'is a lowercase word')
    elif word6.istitle():
        print(word6, 'is a titlecase word')
    else:
        print(word6, 'is others')