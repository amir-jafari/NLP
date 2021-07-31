import spacy

nlp = spacy.load("en_core_web_sm")

print(nlp.vocab.strings.add("text"))
coffee_hash = nlp.vocab.strings["text"]
coffee_string = nlp.vocab.strings[coffee_hash]
print(coffee_string)