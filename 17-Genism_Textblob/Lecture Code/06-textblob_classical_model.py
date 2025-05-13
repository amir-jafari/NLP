#%% --------------------------------------------------------------------------------------------------------------------
from textblob import TextBlob, Word
from textblob.classifiers import NaiveBayesClassifier
#%% --------------------------------------------------------------------------------------------------------------------
# 1. TextBlob Object & Basic Properties
text = "TextBlob makes common NLP tasks very straightforward to implement in Python."
blob = TextBlob(text)
print("Original Text:", blob)
print("Words:", blob.words)
print("Sentences:", blob.sentences)
#%% --------------------------------------------------------------------------------------------------------------------
# 2. Part-of-Speech Tagging & Noun Phrases
print("POS Tags:", blob.tags)
print("Noun Phrases:", blob.noun_phrases)
#%% --------------------------------------------------------------------------------------------------------------------
# 3. Sentiment Analysis
sent1 = TextBlob("I absolutely love this library! It's amazing.")
sent2 = TextBlob("I hate when code doesn't work. It's frustrating.")
print("Sentiment 1:", sent1.sentiment)
print("Sentiment 2:", sent2.sentiment)
#%% --------------------------------------------------------------------------------------------------------------------
# 4. Lemmatization & Spelling Correction
word = Word("geese")
print("Original Word:", word)
print("Lemmatized:", word.lemmatize())
typo = TextBlob("I havv goood spelng.")
print("Before Correction:", typo)
print("After Correction:", typo.correct())
#%% --------------------------------------------------------------------------------------------------------------------
# 5. Word Frequencies & N-grams
text2 = TextBlob("Natural language processing with TextBlob is fun and educational.")
print("Word Counts:", text2.word_counts)
print("2-grams:", text2.ngrams(2))
#%% --------------------------------------------------------------------------------------------------------------------
# 6. Naive Bayes Text Classification
train_data = [
    ("I love this movie. It's fantastic!", 'pos'),
    ("This film was terrible and boring.", 'neg'),
    ("What a great performance by the actors.", 'pos'),
    ("I dislike the storyline; it was dull.", 'neg')
]
clf = NaiveBayesClassifier(train_data)
tests = [
    ("An excellent, thrilling experience.", 'pos'),
    ("Worst movie of the year.", 'neg')
]
for text, _ in tests:
    prob_dist = clf.prob_classify(text)
    print(f"Text: '{text}'")
    print(f"  Label: {prob_dist.max()}")
    print(f"  Probability(positive): {prob_dist.prob('pos'):.2f}")
print(f"Training accuracy: {clf.accuracy(train_data):.2f}")

