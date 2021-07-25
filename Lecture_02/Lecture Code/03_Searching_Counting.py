from nltk.book import text1
from nltk.book import text4
from nltk.book import text6

print(text1.concordance("monstrous"))
print(text1.similar("monstrous"))
print(text1.collocations())
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

print(text6.count("Very"))
print(text6.count('the') / float(len(text6)) * 100)
print(text4.count("bless"))
print(text4[100])
print(text4.index('the'))
print(text4[524])
print(text4.index('men'))
print(text4[0:len(text4)])



