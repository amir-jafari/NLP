from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting to listen to this course my whole life."))
print(classifier("I've been waiting to listen to this terrible course my whole life."))
print(classifier("I am not sure what I am doing but seems I am doing a heck of job"))
print(classifier("How are you doing?.I am not too bad"))

