import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("I love natural language processing.")
lexeme = nlp.vocab["natural"]
print(lexeme.text, lexeme.orth, lexeme.is_alpha)