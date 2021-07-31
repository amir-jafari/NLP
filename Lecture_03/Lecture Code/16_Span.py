from spacy.tokens import Doc, Span
from spacy.lang.en import English
nlp = English()

words = ["Hello", "world", "!"]
spaces = [True, False, False]

doc = Doc(nlp.vocab, words=words, spaces=spaces)
span = Span(doc, 0, 2)
span_with_label = Span(doc, 0, 2, label="GREETING")
doc.ents = [span_with_label]; print(doc.ents)