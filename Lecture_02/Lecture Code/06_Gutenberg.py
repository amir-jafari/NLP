from nltk.corpus import gutenberg
import nltk

print(gutenberg.fileids())
emma = gutenberg.words('austen-emma.txt')
print(len(emma))
emma_Text = nltk.Text(gutenberg.words('austen-emma.txt'))
emma_Text.concordance("surprize")
for fileid in gutenberg.fileids():
   num_chars = len(gutenberg.raw(fileid))
   num_words = len(gutenberg.words(fileid))
   num_sents = len(gutenberg.sents(fileid))
   num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
   print(round(num_chars / num_words), round(num_words / num_sents),
         round(num_words / num_vocab), fileid)

macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt');
print(macbeth_sentences)
print(macbeth_sentences[1116])
longest_len = max(len(s) for s in macbeth_sentences); print(longest_len)
print( [s for s in macbeth_sentences if len(s) == longest_len])