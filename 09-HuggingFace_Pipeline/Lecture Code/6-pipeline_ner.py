from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
print(ner("My name is Amir and I work at GWU in District of Columbia office."))