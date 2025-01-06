from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

# Sample dataset
documents = [
    "Machine learning is amazing",
    "Natural language processing and machine learning go hand in hand",
    "Artificial intelligence drives machine learning",
    "Deep learning is a subset of machine learning",
    "Natural language processing is a field of AI",
]

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)


n_topics = 2  # Number of topics
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)


def print_topics(model, feature_names, n_top_words=5):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print("\n")

print("Topics Identified:")
print_topics(lda, vectorizer.get_feature_names_out())


doc_topic_distribution = lda.transform(X)
doc_topic_df = pd.DataFrame(doc_topic_distribution, columns=[f"Topic {i+1}" for i in range(n_topics)])
doc_topic_df['Dominant Topic'] = doc_topic_df.idxmax(axis=1)
doc_topic_df['Document'] = documents
print("\nDocument-Topic Distribution:")
print(doc_topic_df)
