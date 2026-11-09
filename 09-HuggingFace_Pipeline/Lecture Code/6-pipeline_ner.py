from transformers import pipeline
from transformers.utils import logging

logging.set_verbosity_error()

ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", revision="4c53496",
               aggregation_strategy="simple")
print(ner("My name is Amir and I work at GWU in District of Columbia office."))
