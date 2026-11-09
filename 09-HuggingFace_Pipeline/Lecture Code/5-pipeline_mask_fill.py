from transformers import pipeline
from transformers.utils import logging

logging.set_verbosity_error()

unmasker = pipeline("fill-mask", model="distilbert/distilroberta-base", revision="fb53ab8")
print(unmasker("This session will teach you all about <mask> models.", top_k=2))
print(unmasker("I <mask> my life.", top_k=5))
