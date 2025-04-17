import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.language import Language

@Language.component("component1")
def custom_component1(doc):
    print("Doc length:", len(doc))
    return doc
nlp.add_pipe('component1', name="component-info-1", first=True)
print("Pipeline:", nlp.pipe_names)

@Language.component("component2")
def custom_component2(doc):
    print("Doc length:", len(doc))
    return doc

nlp.add_pipe('component2', name="component-info-2", first=True)
doc = nlp("Hello world!")
print(doc)