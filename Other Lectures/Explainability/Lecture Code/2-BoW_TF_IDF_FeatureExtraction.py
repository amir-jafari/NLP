from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Sample text data
corpus = ["NLP is amazing", "Explainability is crucial in AI", "LIME and SHAP help interpret models"]

# BoW representation
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(corpus)

# TF-IDF representation
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)

# Convert to DataFrame for better readability
bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

print("Bag of Words Representation:\n", bow_df)
print("\nTF-IDF Representation:\n", tfidf_df)
