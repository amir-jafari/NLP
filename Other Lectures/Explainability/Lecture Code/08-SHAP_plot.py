import pandas as pd
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv("imdb_train.csv")
df_test = pd.read_csv("imdb_test.csv")

# Extract text and labels
train_texts = df_train["text"]
train_labels = df_train["label"]

test_texts = df_test["text"]
test_labels = df_test["label"]

vectorizer = TfidfVectorizer(stop_words='english')

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, train_labels)

explainer = shap.Explainer(clf, X_train)

shap_values = explainer(X_test)

#%%
# ==========================================================================
# The Below code is to create
#   1- Waterfall Plot for a single test instance
#   2- Bar Plot (global feature importance across the test set)
# ==========================================================================
shap.plots.waterfall(shap_values[0])

shap.plots.bar(shap_values)
