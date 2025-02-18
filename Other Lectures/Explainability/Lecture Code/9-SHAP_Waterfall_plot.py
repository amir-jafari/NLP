import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1) Sample data (includes both ham and spam)
texts = [
    "This is the Professor Amir's NLP code example!",  # ham
    "Free money now!!!",  # spam
    "Hi John, are we still on for coffee?",  # ham
    "Congratulations, you've won a $100 gift card",  # spam
]

# Labels: 0 = ham, 1 = spam
labels = [0, 1, 0, 1]

# 2) Prepare the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# 3) Fit the vectorizer and transform the text data
X = vectorizer.fit_transform(texts)

# 4) Train a logistic regression model (classifier)
clf = LogisticRegression()
clf.fit(X, labels)

# 5) Build the SHAP explainer for the Logistic Regression model
explainer = shap.Explainer(clf, X)  # Directly pass the sparse matrix for explainer

# 6) Get SHAP values for the input text
shap_values = explainer(X)

# 7) Visualize SHAP values using Waterfall Plot
# For a single prediction (e.g., first text in dataset)
shap.plots.waterfall(shap_values[0])  # This visualizes how each feature contributed to the first prediction

# 8) Visualize SHAP values using Bar Plot (Global feature importance across the dataset)
shap.plots.bar(shap_values)  # This shows a global view of feature importance across all predictions
