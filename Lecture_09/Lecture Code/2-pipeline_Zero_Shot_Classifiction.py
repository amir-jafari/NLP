from transformers import pipeline
classifier = pipeline("zero-shot-classification")

print(classifier("This is a course about the Transformers library",
                 candidate_labels=["education", "politics", "business"],))

classifier("This session is about the machine learning and artifical inteligence",
           candidate_labels=["education", "politics", "business", "data science"],)