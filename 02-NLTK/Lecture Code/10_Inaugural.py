from nltk.corpus import inaugural
import matplotlib.pyplot as plt
import nltk

cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])  # Extract year as the label
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target)
)


years = sorted(set(fileid[:4] for fileid in inaugural.fileids()))
years_numeric = [int(year) for year in years]  # Convert to integers


plt.figure(figsize=(14, 6))
for word in cfd.conditions():
    frequencies = [cfd[word][year] for year in years]
    plt.plot(years_numeric, frequencies, marker='o', label=word)
plt.xticks(years_numeric, [str(year) for year in years_numeric], rotation=45, ha='right')
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.title("Word Frequency in Inaugural Speeches")
plt.legend()
plt.grid(False)
plt.show()
