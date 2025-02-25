# ======================================================================================================================
# Shared Setup - Loading dataset 'amazon_polarity'
# ----------------------------------------------------------------------------------------------------------------------
#%% --------------------------------------------------------------------------------------------------------------------
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer
import pandas as pd

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
#%%
# ======================================================================================================================
# Class_Ex1:
# How can you create a global overview of Explanations using LIME for multiple instances at once?
# LIME is often demonstrated on single-instance explanations.
# However, you might want a summary across a larger dataset (e.g., to see which words are most influential overall).
# One approach is to iterate over many samples, gather each local explanation, and aggregate feature weights.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')



print(20*'-' + 'End Q1' + 20*'-')

# ======================================================================================================================
# Class_Ex2:
# How to compare SHAP Explanations for Two Labels (Logistic Regression Vs. Random Forest)?
# Sometimes you want to see how two model types (e.g., LR vs. RF) weigh features differently for a single instance.
# Step 1: Train both models
# Step 2: Use SHAP to generate explanations
# Step 3: Compare the results
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')


print(20*'-' + 'End Q2' + 20*'-')

# ======================================================================================================================
# Class_Ex3:
# How can we visualize the uncertainty or variance in local explanations from LIME?
# LIME’s random perturbations can produce slightly different explanations each run.
# Step 1: Run LIME multiple times on the same instance
# Step 2: Collect Top Features
# Step 3: Check how stable the explanations are
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')



print(20*'-' + 'End Q3' + 20*'-')

# ======================================================================================================================
# Class_Ex4:
# How can we incorporate domain-specific synonyms or phrases in LIME for text explanations?
# LIME typically handles perturbations by removing words. If you want domain-specific synonyms (perhaps “coach” ↔
# “manager,” “firm” ↔ “company”) so that perturbations are more realistic, you can define a custom function that
# replaces words with synonyms from a small dictionary.
# ----------------------------------------------------------------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')


print(20*'-' + 'End Q4' + 20*'-')
import pandas as pd




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





print(20*'-' + ' End Q1 ' + 20*'-', "\n")

