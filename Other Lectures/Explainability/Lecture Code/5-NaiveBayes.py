from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample text data
corpus = ["NLP is amazing", "Explainability is crucial in AI", "LIME and SHAP help interpret models"]

# TF-IDF representation
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)

# Train Naïve Bayes
nb = MultinomialNB()
nb.fit(X_tfidf, [0, 1, 1])  # Assume binary labels

# Feature importance via log probabilities
feature_probs = np.exp(nb.feature_log_prob_)

# Display word contributions
word_importance = pd.DataFrame(feature_probs, columns=tfidf.get_feature_names_out())
print("\nNaïve Bayes Word Contributions:\n", word_importance)
