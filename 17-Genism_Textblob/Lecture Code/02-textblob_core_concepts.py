#%% --------------------------------------------------------------------------------------------------------------------
from textblob import TextBlob
from textblob import Word
#%% --------------------------------------------------------------------------------------------------------------------
# Create a Textblob
print('=========================================Textblob==============================================================')
wiki = TextBlob("Python is a high-level, general-purpose programming language.")
print(wiki)
#%% --------------------------------------------------------------------------------------------------------------------
# Part-of-speech Tagging
print('=========================================Part-of-speech Tagging================================================')
wiki_tags = wiki.tags
print(wiki_tags)
#%% --------------------------------------------------------------------------------------------------------------------
# Noun Phrase Extraction
wiki_noun_phrases = wiki.noun_phrases
print(wiki_noun_phrases)
#%% --------------------------------------------------------------------------------------------------------------------
# Sentiment Analysis
testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")
print(testimonial)
testimonial_sentiment = testimonial.sentiment
print(testimonial_sentiment)
testimonial_sentiment_polarity = testimonial.sentiment.polarity
print(testimonial_sentiment_polarity)
#%% --------------------------------------------------------------------------------------------------------------------
# Tokenization
zen = TextBlob("Beautiful is better than ugly. "
    "Explicit is better than implicit. "
    "Simple is better than complex."
)
zen_words = zen.words
print(zen_words)
zen_sentences = zen.sentences
print(zen_sentences)
for sentence in zen.sentences:
    print(sentence.sentiment)
#%% --------------------------------------------------------------------------------------------------------------------
# Words Inflection and Lemmatization
sentence = TextBlob("Use 4 spaces per indentation level.")
print(sentence)
sentence_words = sentence.words
print(sentence_words)
sentence_word1 = sentence.words[2].singularize()
print(sentence_word1)
sentence_word2 = sentence.words[-1].pluralize()
print(sentence_word2)
#%% --------------------------------------------------------------------------------------------------------------------
# Words can be lemmatized by calling the lemmatize method.
w1 = Word("octopi")
print(w)
w1_lemmatize = w1.lemmatize()
print(w1_lemmatize)
w2 = Word("went")
print(w2)
w2_lemmatize = w2.lemmatize("v")
print(w2_lemmatize)
#%% --------------------------------------------------------------------------------------------------------------------
