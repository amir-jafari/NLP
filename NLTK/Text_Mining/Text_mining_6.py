# %%%%%%%%%%%%% Natural Language Processing %%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 11 - 8 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Text Mining %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------Some Basics----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
empty = []
nested = [empty, empty, empty]
nested.append('Python')
print(nested)

empty = []
nested = [empty, empty, empty]
nested[0].append('Python')
print(nested)
nested[1]=['test']
print(nested)

# ----------------------------------------------------------------------------------------------------------------------
import nltk

Text = ['cat', '', ['dog'], []]
for i in Text:
    if i:
        print(i)

Text1 = ['cat', 'dog', 'tiger']

for i in Text1:
    if len(i)<4:
        print(i)

text = nltk.corpus.gutenberg.words('milton-paradise.txt')
longest = ''
for word in text:
    if len(word) > len(longest):
        longest = word

print(longest)

# ----------------------------------------------------------------------------------------------------------------------
Text = ['Cat', 'hi', 'Dog', 'me', 'Data', 'Machine', 'Corp', 'no', '.']

All = all(len(w) > 4 for w in Text)
ANY = any(len(w) > 4 for w in Text)

print(All)
print(ANY)
# ----------------------------------------------------------------------------------------------------------------------

Text = ['Cat', 'hi', 'Dog', 'me', 'Data', 'Machine', 'Corp', 'no', '.']
Count = 3
a = []
for i in range(len(Text) - Count +1):
    a.append(Text[i:i + Count])
print(a)