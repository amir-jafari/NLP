from spacy.tokens import Token
import spacy
nlp = spacy.load("en_core_web_sm")

def get_is_color(token):
    colors = ["red", "yellow", "blue"]
    return token.text in colors

Token.set_extension("is_color", getter=get_is_color)

doc = nlp("The sky is blue.")
print(doc[3]._.is_color, "-", doc[3].text)