from nltk.book import text1
from nltk.book import text4

print(text4[100])
print(text4.index('the'))
print(text4[524])
print(text4.index('men'))
print(text4[0:len(text4)])

print(set(text4))
print(sorted(set(text4)))
print(sorted(set(text4)))
print(len(set(text4)))

T1_diversity = float(len(set(text1))) / float(len(text1))
print("The lexical diversity is: ", T1_diversity * 100, "%")
T4_diversity = float(len(set(text4))) / float(len(text4))
print("The lexical diversity is: ", T4_diversity * 100, "%")
