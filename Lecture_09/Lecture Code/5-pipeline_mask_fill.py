from transformers import pipeline

unmasker = pipeline("fill-mask")
print(unmasker("This session will teach you all about <mask> models.", top_k=2))
print(unmasker("I <mask> my life.", top_k=5))