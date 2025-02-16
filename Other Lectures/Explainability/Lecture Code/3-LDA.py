from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Apply LDA (2 topics)
lda = LatentDirichletAllocation(n_components=2, random_state=42)
X_lda = lda.fit_transform(X_tfidf)

# Display top words per topic
def display_topics(model, feature_names, num_top_words=5):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

print("\nLDA Topics:")
display_topics(lda, vectorizer.get_feature_names_out())
