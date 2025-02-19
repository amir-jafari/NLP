# ==========================================================================
#
# User Case 1 - Sentiment Analysis (SHAP Version)
#
# ==========================================================================

# ==========================================================================
# The dataset 'IMDb' is used in this user case. Small Part of 'IMDb' is selected
# for sentiment Analysis, and the goal is to classify reviews as positive or
# negative.
#
# This py file is to show how the SHAP package can be used in the Sentiment
# Analysis, similar to LIME but with SHAP.
# ==========================================================================

# ==========================================================================
# Step 1 - Load Dataset
# ==========================================================================
import pandas as pd

df_train = pd.read_csv('imdb_train.csv')
df_test = pd.read_csv('imdb_test.csv')

# %%
# ==========================================================================
# Step 2 - Train Pipeline
# ==========================================================================
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import shap

pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression(max_iter=200)
)
pipeline.fit(df_train['text'], df_train['label'])

# ==========================================================================
# Step 3 - Pick One Example
# ==========================================================================
test_idx = 0
test_text = df_test['text'].iloc[test_idx]
true_label = df_test['label'].iloc[test_idx]

print(f"\nExplaining test instance #{test_idx}")
print("-----------------------------------")
print(f"True Label: {true_label}")
print("Review Snippet:", test_text[:200], "...")

# ==========================================================================
# Step 4 - SHAP Explanation
# ==========================================================================
def predict_proba_text(texts):
    return pipeline.predict_proba(texts)

explainer = shap.Explainer(
    predict_proba_text,
    masker=shap.maskers.Text("word"),
    output_names=["Negative", "Positive"]  # Just naming the two classes
)

# can pass a list of multiple, but here we just pass the single
shap_values = explainer([test_text])

# ==========================================================================
# Step 5 - Show Explanation
# ==========================================================================

print("\n=== SHAP Explanation ===")

# Console-based approach
tokens = shap_values[0].data

# 1) Determine the class that the model predicted for this instance
predicted_class = pipeline.predict([test_text])[0]
# 2) Extract SHAP values for that single class (=> 1D array)
scores = shap_values[0].values[:, predicted_class]

# Sort tokens by absolute SHAP value (most impactful words first)
indices_sorted = sorted(range(len(scores)), key=lambda i: abs(scores[i]), reverse=True)

print("\nTop word contributions (SHAP):")
for i in indices_sorted[:10]:
    print(f"{tokens[i]:<15} => SHAP value = {scores[i]:.3f}")
