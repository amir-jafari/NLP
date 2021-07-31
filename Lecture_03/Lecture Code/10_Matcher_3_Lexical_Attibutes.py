import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
pattern = [{"IS_DIGIT": True}, {"LOWER": "fifa"}, {"LOWER": "world"},
           {"LOWER": "cup"},  {"IS_PUNCT": True}]

matcher.add("FIFA", [pattern])
doc = nlp("2018 FIFA World Cup: France won!")
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)