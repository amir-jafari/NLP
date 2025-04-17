import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [{"LEMMA": "buy"}, {"POS": "DET", "OP": "?"},  {"POS": "NOUN"}]
matcher.add("Other", [pattern])

doc = nlp("I bought a smartphone. Now I'm buying apps.")
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)