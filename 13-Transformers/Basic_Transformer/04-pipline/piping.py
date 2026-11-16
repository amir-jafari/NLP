import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
# ----------------------------------------------------------------------------------------------------------------------
# 1) Sentiment analysis with a ready-made pipeline
nlp = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(nlp("If you sometimes like to go to the movies to have fun , Wasabi is a good place to start ."),
      "If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .")
print(nlp("Effective but too-tepid biopic."), "Effective but too-tepid biopic.")
# ----------------------------------------------------------------------------------------------------------------------
# 2) Paraphrase detection "by hand": tokenizer -> model -> softmax  (what a pipeline does inside)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")
classes = ["not paraphrase", "is paraphrase"]
sequence_A = "The DVD-CCA then appealed to the state Supreme Court."
sequence_B = "The DVD CCA appealed that decision to the U.S. Supreme Court."

paraphrase = tokenizer(sequence_A, sequence_B, return_tensors="pt")
with torch.no_grad():
    paraphrase_classification_logits = model(**paraphrase).logits
paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1)[0].tolist()
print(sequence_B, "should be a paraphrase")
for i in range(len(classes)):
    print(f"{classes[i]}: {round(paraphrase_results[i] * 100)}%")
# ----------------------------------------------------------------------------------------------------------------------
# 3) Named entity recognition
nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very " \
           "close to the Manhattan Bridge which is visible from the window."
print(nlp(sequence))
# ----------------------------------------------------------------------------------------------------------------------
# 4) Translation with T5 (task is given as a text prefix)
t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
t5 = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
inputs = t5_tokenizer("translate English to French: The car could not go in the garage because it was too big.",
                      return_tensors="pt")
print(t5_tokenizer.decode(t5.generate(**inputs, max_length=40)[0], skip_special_tokens=True))
