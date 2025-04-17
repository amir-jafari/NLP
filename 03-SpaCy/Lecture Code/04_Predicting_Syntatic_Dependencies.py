# https://spacy.io/usage/linguistic-features
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
doc = nlp("She ate the pizza")
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)


analyzer = SentimentIntensityAnalyzer()
text = "This is the best movie I have ever seen. Amazing experience!"
doc = nlp(text)
sentiment = analyzer.polarity_scores(doc.text)
print(sentiment)
