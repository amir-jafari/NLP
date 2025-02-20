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
# Step 2 - Train Pipelines
# ==========================================================================

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import shap

# Pipeline for Logistic Regression
pipeline_lr = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression(max_iter=200)
)
pipeline_lr.fit(df_train['text'], df_train['label'])

# Pipeline for Random Forest
pipeline_rf = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
pipeline_rf.fit(df_train['text'], df_train['label'])

# Pipeline for XGBoost
pipeline_xgb = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss')
)
pipeline_xgb.fit(df_train['text'], df_train['label'])

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
# Step 4 - SHAP Explanations
# ==========================================================================

# -- Logistic Regression --
def predict_proba_text_lr(texts):
    return pipeline_lr.predict_proba(texts)

explainer_lr = shap.Explainer(
    predict_proba_text_lr,
    masker=shap.maskers.Text("word"),
    output_names=["Negative", "Positive"]
)
shap_values_lr = explainer_lr([test_text])

# -- Random Forest --
def predict_proba_text_rf(texts):
    return pipeline_rf.predict_proba(texts)

explainer_rf = shap.Explainer(
    predict_proba_text_rf,
    masker=shap.maskers.Text("word"),
    output_names=["Negative", "Positive"]
)
shap_values_rf = explainer_rf([test_text])

# -- XGBoost --
def predict_proba_text_xgb(texts):
    return pipeline_xgb.predict_proba(texts)

explainer_xgb = shap.Explainer(
    predict_proba_text_xgb,
    masker=shap.maskers.Text("word"),
    output_names=["Negative", "Positive"]
)
shap_values_xgb = explainer_xgb([test_text])

# ==========================================================================
# Step 5 - Show Explanation
# ==========================================================================
print("\n=== SHAP Explanations ===")

# --- Logistic Regression ---
tokens_lr = shap_values_lr[0].data
predicted_class_lr = pipeline_lr.predict([test_text])[0]
scores_lr = shap_values_lr[0].values[:, predicted_class_lr]

indices_sorted_lr = sorted(
    range(len(scores_lr)),
    key=lambda i: abs(scores_lr[i]),
    reverse=True
)

print("\nTop word contributions (SHAP) - Logistic Regression:")
for i in indices_sorted_lr[:10]:
    print(f"{tokens_lr[i]:<15} => SHAP value = {scores_lr[i]:.3f}")

# --- Random Forest ---
tokens_rf = shap_values_rf[0].data
predicted_class_rf = pipeline_rf.predict([test_text])[0]
scores_rf = shap_values_rf[0].values[:, predicted_class_rf]

indices_sorted_rf = sorted(
    range(len(scores_rf)),
    key=lambda i: abs(scores_rf[i]),
    reverse=True
)

print("\nTop word contributions (SHAP) - Random Forest:")
for i in indices_sorted_rf[:10]:
    print(f"{tokens_rf[i]:<15} => SHAP value = {scores_rf[i]:.3f}")

# --- XGBoost ---
tokens_xgb = shap_values_xgb[0].data
predicted_class_xgb = pipeline_xgb.predict([test_text])[0]
scores_xgb = shap_values_xgb[0].values[:, predicted_class_xgb]

indices_sorted_xgb = sorted(
    range(len(scores_xgb)),
    key=lambda i: abs(scores_xgb[i]),
    reverse=True
)

print("\nTop word contributions (SHAP) - XGBoost:")
for i in indices_sorted_xgb[:10]:
    print(f"{tokens_xgb[i]:<15} => SHAP value = {scores_xgb[i]:.3f}")

