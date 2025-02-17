import shap
from transformers import pipeline

# a text-classification pipeline
model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
pipe = pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None)

# set the texts variable
texts = ["This is the Professor Amir's NLP code example!"]

# 3) Build the SHAP explainer
explainer = shap.Explainer(pipe)
shap_output = explainer(texts)

print("shap_output.values shape:", shap_output.values.shape)
# e.g. (1, 14, 2) => 1 sample, 14 tokens, 2 classes (ham/spam).

shap_spam_class = shap_output[..., 1]
shap_single = shap_spam_class[0]

shap_single.data = [
    "[EMPTY]" if isinstance(token, str) and token == "" else token
    for token in shap_single.data
]

shap.plots.waterfall(shap_single, max_display=10)

shap.plots.bar(shap_single, max_display=10)
