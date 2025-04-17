import spacy
nlp = spacy.load("en_core_web_md")

doc = nlp("I like pizza")
token = nlp("soap")[0]

print(doc.similarity(token))
span = nlp("I like pizza and pasta")[2:5]
doc = nlp("McDonalds sells burgers")

print(span.similarity(doc))