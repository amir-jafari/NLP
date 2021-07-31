import spacy
nlp = spacy.load("en_core_web_sm")

# docs = [nlp(text) for text in LOTS_OF_TEXTS]---Slow
# docs = list(nlp.pipe(LOTS_OF_TEXTS))---Fast

# doc = nlp("Hello world")
# doc = nlp.make_doc("Hello world!")

text = 'I love performance'
with nlp.disable_pipes("tagger", "parser"):
    doc = nlp(text)
    print(doc.text)