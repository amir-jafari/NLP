# ==========================================================================
#
# User Case 1 - Sentiment Analysis
#
# ==========================================================================

# ==========================================================================
# The dataset 'IMDb' is used in this user case. Small Part of 'IMDb' is selected
# for sentiment Analysis, and the goal is to classify reviews as positive or
# negative.
#
# This py file is to show how the LIME package can be used in the Sentiment
# Analysis.
# ==========================================================================

# ==========================================================================
# Step 1 - Load Dataset
# ==========================================================================

import pandas as pd

df_train = pd.read_csv('imdb_train.csv')
df_test = pd.read_csv('imdb_test.csv')


#%%
# ==========================================================================
# Step 2 - Train Pipeline
# ==========================================================================
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Official LIME
from lime.lime_text import LimeTextExplainer

pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression(max_iter=200)
)
pipeline.fit(df_train['text'], df_train['label'])

# ==========================================================================
# Step 3 - Pick One Example
# ==========================================================================
test_idx = 0
test_text  = df_test['text'].iloc[test_idx]
true_label = df_test['label'].iloc[test_idx]

print(f"\nExplaining test instance #{test_idx}")
print("-----------------------------------")
print(f"True Label: {true_label}")
print("Review Snippet:", test_text[:200], "...")

# ==========================================================================
# Step 4 - LIME Explanation
# ==========================================================================
explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
exp = explainer.explain_instance(
    test_text,
    pipeline.predict_proba,
    num_features=10
)

# ==========================================================================
# Step 5 - Show Explanation
# ==========================================================================
print("\nTop word contributions (LIME):")
for word, weight in exp.as_list():
    print(f"{word:<15} weight={weight:.3f}")


#%%
print(df_train['label'].value_counts())
