from sklearn.linear_model import LogisticRegression
import shap
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
corpus = ["NLP is amazing", "Explainability is crucial in AI", "LIME and SHAP help interpret models"]


# TF-IDF representation
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)

# Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_tfidf, [0, 1, 1])

# Apply SHAP
explainer = shap.Explainer(log_reg, X_tfidf)
shap_values = explainer(X_tfidf)

# Visualize feature importance
shap.summary_plot(shap_values, X_tfidf, feature_names=tfidf.get_feature_names_out())
