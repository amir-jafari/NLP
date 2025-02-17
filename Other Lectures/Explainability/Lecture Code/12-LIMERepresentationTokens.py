"""
A minimal demo of LIME token-level explanations on a small spam/ham dataset.
We:
1) Create 4 short text samples (2 spam, 2 ham).
2) Convert to TF-IDF and train a logistic regression.
3) Use lime.lime_text.LimeTextExplainer to show local token attributions.
4) Display or save an HTML explanation.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer
import webbrowser

# ---------------------------------------------------------------------
# 1) Tiny spam/ham dataset
# ---------------------------------------------------------------------
spam_ham_data = [
    ("Congratulations! You've won a free cruise!", 1),   # spam
    ("Hurry up! Last chance to claim your prize!", 1),   # spam
    ("Are we meeting tomorrow about the project?", 0),   # ham
    ("Let me know if you're free for lunch", 0)          # ham
]

X = [pair[0] for pair in spam_ham_data]
y = [pair[1] for pair in spam_ham_data]

# ---------------------------------------------------------------------
# 2) Vectorize and train a simple classifier
# ---------------------------------------------------------------------
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

clf = LogisticRegression()
clf.fit(X_tfidf, y)

# ---------------------------------------------------------------------
# 3) Pick a new message
# ---------------------------------------------------------------------
text_instance = "You've won a FREE ticket! Please claim now."

# ---------------------------------------------------------------------
# 4) LIME explanation
# ---------------------------------------------------------------------
explainer = LimeTextExplainer(class_names=["ham","spam"])

# The classifier_fn must accept a list of strings, return probs shape [n,2]
def classifier_fn(texts):
    # Transform text using the same TF-IDF
    text_tfidf = tfidf.transform(texts)
    # Return predicted probability for each text
    return clf.predict_proba(text_tfidf)

# Explain instance with top 5 tokens
exp = explainer.explain_instance(
    text_instance,
    classifier_fn=classifier_fn,
    num_features=5
)

# ---------------------------------------------------------------------
# 5) Save the explanation to an HTML file
# ---------------------------------------------------------------------
html_explanation = exp.as_html()
with open("lime_explanation.html", "w", encoding="utf-8") as f:
    f.write(html_explanation)

webbrowser.open("lime_explanation.html")

