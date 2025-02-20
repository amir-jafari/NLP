# -*- coding: utf-8 -*-
"""
Sentiment Analysis with LIME Explanations (IMDb Movie Reviews)

This script demonstrates how to:
1. Load a small subset of the IMDb dataset for binary classification (positive/negative).
2. Train three different models (Logistic Regression, RandomForest, and XGBoost) in pipelines
   that include TF-IDF vectorization.
3. Use the LIME package to interpret model predictions by identifying
   which words contribute most to a specific prediction.
4. Compare explanatory word importance across multiple models.
5. Evaluate each model's overall accuracy to give additional context.
"""

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# LIME
from lime.lime_text import LimeTextExplainer

# Metrics
from sklearn.metrics import accuracy_score

# ==========================================================================
# Step 1 - Load Dataset
# ==========================================================================
df_train = pd.read_csv('imdb_train.csv')
df_test = pd.read_csv('imdb_test.csv')

# Split into features and labels for clarity
X_train, y_train = df_train['text'], df_train['label']
X_test, y_test   = df_test['text'], df_test['label']

# ==========================================================================
# Step 2 - Train Pipelines
# ==========================================================================
# Define pipelines for three different classifiers (LogisticRegression, RandomForest, XGBoost)

pipeline_lr = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression(max_iter=200)
)

pipeline_rf = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

pipeline_xgb = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
)

# Fit each pipeline to the training set
pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)
pipeline_xgb.fit(X_train, y_train)

# ==========================================================================
# Step 3 - Evaluate Each Model
# ==========================================================================
# We add evaluation to see how well each model performs overall on the test set

pred_lr  = pipeline_lr.predict(X_test)
pred_rf  = pipeline_rf.predict(X_test)
pred_xgb = pipeline_xgb.predict(X_test)

acc_lr  = accuracy_score(y_test, pred_lr)
acc_rf  = accuracy_score(y_test, pred_rf)
acc_xgb = accuracy_score(y_test, pred_xgb)

print("Model Accuracy on Test Set:")
print("---------------------------")
print(f"Logistic Regression: {acc_lr:.3f}")
print(f"Random Forest:       {acc_rf:.3f}")
print(f"XGBoost:            {acc_xgb:.3f}")

# ==========================================================================
# Step 4 - Pick One Example from the Test Set
# ==========================================================================
# We'll use a single review to demonstrate LIME's explanation of the predicted label

test_idx = 0
test_text  = X_test.iloc[test_idx]
true_label = y_test.iloc[test_idx]

print(f"\nExplaining test instance #{test_idx}")
print("-----------------------------------")
print(f"True Label: {true_label}")
print("Review Snippet:", test_text[:200], "...")

# ==========================================================================
# Step 5 - LIME Explanation
# ==========================================================================
# Initialize a LimeTextExplainer. We explicitly pass class names in the same
# order that the models predict (Negative, then Positive).

explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

exp_lr = explainer.explain_instance(
    test_text,
    pipeline_lr.predict_proba,
    num_features=10
)
exp_rf = explainer.explain_instance(
    test_text,
    pipeline_rf.predict_proba,
    num_features=10
)
exp_xgb = explainer.explain_instance(
    test_text,
    pipeline_xgb.predict_proba,
    num_features=10
)

# ==========================================================================
# Step 6 - Show Word Contributions for Each Model
# ==========================================================================
print("\nTop word contributions (LIME) - Logistic Regression:")
for word, weight in exp_lr.as_list():
    print(f"{word:<15} weight={weight:.3f}")

print("\nTop word contributions (LIME) - Random Forest:")
for word, weight in exp_rf.as_list():
    print(f"{word:<15} weight={weight:.3f}")

print("\nTop word contributions (LIME) - XGBoost:")
for word, weight in exp_xgb.as_list():
    print(f"{word:<15} weight={weight:.3f}")
