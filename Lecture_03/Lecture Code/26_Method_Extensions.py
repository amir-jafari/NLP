from spacy.tokens import Doc
import spacy
nlp = spacy.load("en_core_web_sm")

def has_token(doc, token_text):
    in_doc = token_text in [token.text for token in doc]
    return in_doc

Doc.set_extension("has_token", method=has_token)

doc = nlp("The sky is blue.")
print(doc._.has_token("blue"), "- blue")
print(doc._.has_token("cloud"), "- cloud")