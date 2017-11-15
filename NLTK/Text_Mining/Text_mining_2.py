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

texts()

print(dir(text1))


print(text1.vocab())
print(text2.vocab())

print(type(text1))
print(len(text1))
# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Searching and find Simallar words-----------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

text1.concordance("monstrous")
text1.similar("monstrous")


text2.similar("monstrous")
text2.common_contexts(["monstrous", "very"])

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Collocations--------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


text1.collocations()
text2.collocations()

text3.collocations()
text4.collocations()

text5.collocations()
text6.collocations()

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Word Count----------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

text6.count("Very")
text6.count('the') / float(len(text6)) * 100
text6.count('?') / float(len(text6) * 100)
text6.count('.') / float(len(text6) * 100)
text6.count("a") / float(len(text6) * 100)
text6.count(',') / float(len(text6) * 100)

text4.count("bless")
text4.count('the') / float(len(text4)) * 100
text4.count('?') / float(len(text4) * 100)
text4.count('.') / float(len(text4) * 100)
text4.count("a") / float(len(text4) * 100)
text4.count(',') / float(len(text4) * 100)

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Indexing------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
text4[100]
text4.index('the')
text4[524]
text4.index('men')
text4[0:len(text4)]

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Unique Words--------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

set(text3)
sorted(set(text3))
sor = sorted(set(text3))
len(set(text3))

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Lexical Diversity---------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


T1_diversity = float(len(set(text1))) / float(len(text1))
T2_diversity = float(len(set(text2))) / float(len(text2))
T3_diversity = float(len(set(text3))) / float(len(text3))
T4_diversity = float(len(set(text4))) / float(len(text4))
T5_diversity = float(len(set(text5))) / float(len(text5))
T6_diversity = float(len(set(text6))) / float(len(text6))


print("The lexical diversity is: ", T1_diversity * 100, "%")
print("The lexical diversity is: ", T2_diversity * 100, "%")
print("The lexical diversity is: ", T3_diversity * 100, "%")
print("The lexical diversity is: ", T4_diversity * 100, "%")
print("The lexical diversity is: ", T5_diversity * 100, "%")
print("The lexical diversity is: ", T6_diversity * 100, "%")

