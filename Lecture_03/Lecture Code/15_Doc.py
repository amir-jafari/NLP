from spacy.lang.en import English
nlp = English()

from spacy.tokens import Doc

words = ["Hello", "world", "!"]
spaces = [True, False, False]

doc = Doc(nlp.vocab, words=words, spaces=spaces)