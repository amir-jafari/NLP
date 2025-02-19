# ==========================================================================
#
# User Case 2 - Spam Detection (LIME)
#
# ==========================================================================

# ==========================================================================
# The dataset 'SMS Spam Collection' (url:
# https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
# is used in this user case.
#
# This py file is to show how the LIME package can be used in Spam
# Detection with three different models: Logistic Regression, XGBoost,
# and Random Forest.
# ==========================================================================

# %%
# ==========================================================================
# Step 1 - Load Dataset
# ==========================================================================
import pandas as pd

df = pd.read_csv("spam.csv", encoding="latin-1")

print(df.head(5))

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(df['label'].value_counts())

# %%
# ==========================================================================
# Step 2 - Necessary Packages
# ==========================================================================
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# XGBoost and Random Forest
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# LIME for text
from lime.lime_text import LimeTextExplainer

# %%
# ==========================================================================
# Step 3 - Split the dataset
# ==========================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.3, random_state=42
)

# %%
# ==========================================================================
# Step 4 - Build Pipelines
#   1) Logistic Regression
#   2) XGBoost
#   3) Random Forest
# ==========================================================================
pipeline_lr = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression(max_iter=200)
)

pipeline_xgb = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss')
)

pipeline_rf = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# %%
# ==========================================================================
# Step 5 - Train & Evaluate
# ==========================================================================
# Train
pipeline_lr.fit(X_train, y_train)
pipeline_xgb.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)

# Evaluate each model
acc_lr = pipeline_lr.score(X_test, y_test)
acc_xgb = pipeline_xgb.score(X_test, y_test)
acc_rf = pipeline_rf.score(X_test, y_test)

print("=== Test Accuracy ===")
print(f"Logistic Regression: {acc_lr:.3f}")
print(f"XGBoost           : {acc_xgb:.3f}")
print(f"Random Forest     : {acc_rf:.3f}")

# %%
# ==========================================================================
# Step 6 - LIME Explanation on One Test Example
# ==========================================================================
test_idx = 0
test_text  = X_test.iloc[test_idx]
true_label = y_test.iloc[test_idx]

print(f"\nExplaining test instance #{test_idx}")
print("-----------------------------------")
print("True Label:", true_label)
print("Text:", test_text[:200], "...")

explainer = LimeTextExplainer(class_names=["ham","spam"])

# 1) LIME for Logistic Regression
exp_lr = explainer.explain_instance(
    test_text,
    pipeline_lr.predict_proba,  # black-box function
    num_features=6
)
print("\n=== LIME Explanation (Logistic Regression) ===")
for feature, weight in exp_lr.as_list():
    print(f"{feature:<15} weight={weight:.3f}")

# 2) LIME for XGBoost
exp_xgb = explainer.explain_instance(
    test_text,
    pipeline_xgb.predict_proba,
    num_features=6
)
print("\n=== LIME Explanation (XGBoost) ===")
for feature, weight in exp_xgb.as_list():
    print(f"{feature:<15} weight={weight:.3f}")

# 3) LIME for Random Forest
exp_rf = explainer.explain_instance(
    test_text,
    pipeline_rf.predict_proba,
    num_features=6
)
print("\n=== LIME Explanation (Random Forest) ===")
for feature, weight in exp_rf.as_list():
    print(f"{feature:<15} weight={weight:.3f}")
