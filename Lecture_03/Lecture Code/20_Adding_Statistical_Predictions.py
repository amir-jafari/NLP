from spacy.matcher import Matcher
import spacy
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

matcher.add("CAR", [[{"LOWER": "golden"}, {"LOWER": "car"}]])
doc = nlp("I have a Golden Car")
for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print("Matched span:", span.text)
    # Get the span's root token and root head token
    print("Root token:", span.root.text)
    print("Root head token:", span.root.head.text)
    # Get the previous token and its POS tag
    print("Previous token:", doc[start - 1].text, doc[start - 1].pos_)
