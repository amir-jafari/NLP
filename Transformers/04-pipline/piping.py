from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from transformers import pipeline
# ----------------------------------------------------------------------------------------------------------------------
nlp = pipeline("sentiment-analysis")
print(nlp("If you sometimes like to go to the movies to have fun , Wasabi is a good place to start ."),
      "If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .")
print(nlp("Effective but too-tepid biopic."),"Effective but too-tepid biopic.")
# ----------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")
classes = ["not paraphrase", "is paraphrase"]
sequence_A = "The DVD-CCA then appealed to the state Supreme Court."
sequence_B = "The DVD CCA appealed that decision to the U.S. Supreme Court."

paraphrase = tokenizer.encode_plus(sequence_A, sequence_B, return_tensors="tf")
paraphrase_classification_logits = model(paraphrase)[0]
paraphrase_results = tf.nn.softmax(paraphrase_classification_logits, axis=1).numpy()[0]
print(sequence_B, "should be a paraphrase")
for i in range(len(classes)):
    print(f"{classes[i]}: {round(paraphrase_results[i] * 100)}%")
# ----------------------------------------------------------------------------------------------------------------------
nlp = pipeline("ner")
sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
           "close to the Manhattan Bridge which is visible from the window."
print(nlp(sequence))
# ----------------------------------------------------------------------------------------------------------------------
translator = pipeline("translation_en_to_fr")
print(translator("The car could not go in the garage because it was too big.", max_length=40))

