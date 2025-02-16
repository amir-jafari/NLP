import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Sample corpus
corpus = [
    "NLP is amazing and explainability is crucial",
    "SHAP and LIME help interpret AI models",
    "Topic modeling is useful for NLP tasks",
    "Dimensionality reduction improves feature selection",
    "LDA and LSA are different topic modeling techniques"
]

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(corpus)

# Apply LSA (SVD with 2 components)
lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X_tfidf)

# Display top words for each topic
terms = vectorizer.get_feature_names_out()
components_df = pd.DataFrame(lsa.components_, index=["Topic 1", "Topic 2"], columns=terms)

print("\nLSA Topics (Top Words per Component):")
print(components_df)
