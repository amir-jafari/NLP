import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)

pattern = nlp("Golden Car")
matcher.add("CAR", [pattern])
doc = nlp("I have a Golden Car")

for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print("Matched span:", span.text)