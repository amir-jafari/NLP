# ======================================================================================================================
# Shared Setup - Loading dataset 'amazon_polarity'
# ----------------------------------------------------------------------------------------------------------------------
#%% --------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer
import shap
import webbrowser
from IPython.display import HTML, display
import random

dataset = load_dataset("amazon_polarity")

train_dataset = dataset["train"].select(range(10000))
test_dataset = dataset["test"].select(range(2000))
train_texts = train_dataset["content"]
train_labels = train_dataset["label"]
test_texts = test_dataset["content"]
test_labels = test_dataset["label"]
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)
preds = clf.predict(X_test)
acc = accuracy_score(test_labels, preds)
class_names = ["Negative", "Positive"]

def predict_proba(text_list):
    return clf.predict_proba(vectorizer.transform(text_list))
#%%
# ======================================================================================================================
# Class_Ex1:
# How can you create a global overview of Explanations using LIME for multiple instances at once?
# LIME is often demonstrated on single-instance explanations.
# However, you might want a summary across a larger dataset (e.g., to see which words are most influential overall).
# One approach is to iterate over many samples, gather each local explanation, and aggregate feature weights.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')
explainer = LimeTextExplainer(class_names=class_names)
indices = list(range(len(test_texts)))
random.shuffle(indices)
sample_indices = indices[:10]

explanations = []
for i in sample_indices:
    text_instance = test_texts[i]
    pred_label = clf.predict(vectorizer.transform([text_instance]))[0]
    exp = explainer.explain_instance(text_instance, classifier_fn=predict_proba, num_features=5, labels=[0,1])
    for word, weight in exp.as_list(label=pred_label):
        explanations.append((word, weight))

df = pd.DataFrame(explanations, columns=["word", "weight"])
grouped = df.groupby("word")["weight"].mean().sort_values(ascending=False)
print(grouped.head(10))

print(20*'-' + 'End Q1' + 20*'-')
#%%
# ======================================================================================================================
# Class_Ex2:
# How to compare SHAP Explanations for Two Labels (Logistic Regression Vs. Random Forest)?
# Sometimes you want to see how two model types (e.g., LR vs. RF) weigh features differently for a single instance.
# Step 1: Train both models
# Step 2: Use SHAP to generate explanations
# Step 3: Compare the results
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')
rf_clf = RandomForestClassifier(n_estimators=50)
rf_clf.fit(X_train, train_labels)

def predict_proba_rf(texts):
    return rf_clf.predict_proba(vectorizer.transform(texts))

text_masker = shap.maskers.Text(r"\W+")

explainer_lr = shap.Explainer(predict_proba, masker=text_masker)
explainer_rf = shap.Explainer(predict_proba_rf, masker=text_masker)

idx = 0
text_instance = test_texts[idx]

shap_values_lr = explainer_lr([text_instance])
shap_values_rf = explainer_rf([text_instance])

html_str_lr = shap.plots.text(shap_values_lr[0], display=False)
html_str_rf = shap.plots.text(shap_values_rf[0], display=False)

lr_file = "lr_explanation.html"
rf_file = "rf_explanation.html"

with open("lr_explanation.html", "w", encoding="utf-8") as f:
    f.write(html_str_lr)
with open("rf_explanation.html", "w", encoding="utf-8") as f:
    f.write(html_str_rf)

webbrowser.open_new_tab(lr_file)
webbrowser.open_new_tab(rf_file)

print(20*'-' + 'End Q2' + 20*'-')
#%%
# ======================================================================================================================
# Class_Ex3:
# How can we visualize the uncertainty or variance in local explanations from LIME?
# LIME’s random perturbations can produce slightly different explanations each run.
# Step 1: Run LIME multiple times on the same instance
# Step 2: Collect Top Features
# Step 3: Check how stable the explanations are
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')
text_instance = test_texts[0]
results = []

for run in range(5):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text_instance, classifier_fn=predict_proba, num_features=5)
    pred_label = clf.predict(vectorizer.transform([text_instance]))[0]
    for word, weight in exp.as_list(label=pred_label):
        results.append((run, word, weight))

df_runs = pd.DataFrame(results, columns=["run","word","weight"])
print(df_runs.groupby("word")["weight"].agg(["mean","std"]).sort_values("mean", ascending=False))

print(20*'-' + 'End Q3' + 20*'-')
#%%
# ======================================================================================================================
# Class_Ex4:
# How can we incorporate domain-specific synonyms or phrases in LIME for text explanations?
# LIME typically handles perturbations by removing words. If you want domain-specific synonyms (perhaps “coach” ↔
# “manager,” “firm” ↔ “company”) so that perturbations are more realistic, you can define a custom function that
# replaces words with synonyms from a small dictionary.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')
domain_synonyms = {
    "great": ["wonderful","excellent","fantastic"],
    "bad": ["terrible","awful"],
    "refund": ["return","reimbursement"],
    "shipping": ["delivery","postage"]
}

def domain_aware_perturbation(text):
    words = text.split()
    new_words = []
    for w in words:
        base = w.lower().strip(".,!?")
        if base in domain_synonyms:
            new_words.append(random.choice(domain_synonyms[base]))
        else:
            new_words.append(w)
    return " ".join(new_words)

explainer_custom = LimeTextExplainer(class_names=class_names)
sample_text = "This product was great but the shipping was bad."
exp_custom = explainer_custom.explain_instance(sample_text, classifier_fn=predict_proba, num_features=5)

print(exp_custom.as_list())
print(20*'-' + 'End Q4' + 20*'-')
#%%
# ======================================================================================================================
# Class_Ex5:
# How do we integrate partial dependence plots (PDP) with LIME/SHAP for text features?
# PDPs are typically for numeric features. With text, you can define “numeric proxies,” such as:
# Text length
# Number of uppercase words
# Sentiment score
# Then see how changing that proxy affects the model output.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + ' Begin Q1 ' + 20*'-')
def text_length(t):
    return len(t.split())

X_len_train = np.array([text_length(txt) for txt in train_texts]).reshape(-1,1)
X_len_test  = np.array([text_length(txt) for txt in test_texts]).reshape(-1,1)

clf_len = LogisticRegression().fit(X_len_train, train_labels)

lengths_to_test = np.linspace(1, 200, 20)
probs = []
for l in lengths_to_test:
    p = clf_len.predict_proba([[l]])[0]
    probs.append(p[1])

plt.plot(lengths_to_test, probs, label="Probability(Positive)")
plt.xlabel("Review Text Length (words)")
plt.ylabel("Prob(Positive)")
plt.title("Partial Dependence of Review Length (Amazon Polarity)")
plt.legend()
plt.show()
print(20*'-' + ' End Q1 ' + 20*'-', "\n")

