import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("I have a banana")
print(doc[3].vector)