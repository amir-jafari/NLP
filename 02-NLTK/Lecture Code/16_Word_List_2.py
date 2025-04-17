import nltk
import matplotlib.pyplot as plt

names = nltk.corpus.names
print(names.fileids())

male_names = names.words('male.txt')
female_names = names.words('female.txt')

print([w for w in male_names if w in female_names])


cfd = nltk.ConditionalFreqDist(
    (fileid, name[-1].lower())
    for fileid in names.fileids()
    for name in names.words(fileid)
)


plt.figure(figsize=(14, 6))
cfd.plot()
letters = sorted(set(name[-1].lower() for name in male_names + female_names if name[-1].isalpha()))
plt.xticks(ticks=range(len(letters)), labels=[letter.upper() for letter in letters], rotation=45, ha='right')
plt.xlabel("Last Letter of Name")
plt.ylabel("Frequency")
plt.title("Frequency of Last Letters in Male and Female Names")
plt.grid(False)
plt.tight_layout()
plt.show()