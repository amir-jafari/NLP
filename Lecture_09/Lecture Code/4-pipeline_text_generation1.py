from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
print(generator("In this course, we will teach you how to"))
print(generator("I am tired of listening to this brownbag session about natural language processing.",
          num_return_sequences = 1, max_length  = 100 ))