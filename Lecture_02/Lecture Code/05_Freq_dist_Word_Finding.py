from nltk.book import text1
from nltk.book import text4
from nltk import FreqDist
import nltk
Freq_Dist = FreqDist(text1)
print(Freq_Dist)
print(Freq_Dist.most_common(10))
print(Freq_Dist['his'])
Freq_Dist.plot(50, cumulative = False)
Freq_Dist.plot(50, cumulative = True)
Freq_Dist.hapaxes()
Once_happend= Freq_Dist.hapaxes() ; print(Once_happend)
print(text4.count('america') / float(len(text4) * 100))

Value_set = set(text1)
long_words = [words for words in Value_set if len(words) > 17]
print(sorted(long_words))
my_text = ["Here", "are", "some", "words", "that", "are", "in", "a", "list"]
vocab = sorted(set(my_text)) ; print(vocab)
word_freq = nltk.FreqDist(my_text); print(word_freq.most_common(5))