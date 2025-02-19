# ==========================================================================
#
# 20 Newsgroups (SHAP) with Logistic Regression, Random Forest, XGBoost
# (Numeric approach to avoid text mismatch)
#
# ==========================================================================
# We train 3 models on TF–IDF features from 20 Newsgroups
# and use SHAP KernelExplainer with numeric arrays.
# This way, we don't run into the "TypeError: cannot use a string pattern
# on a bytes-like object" mismatch between text vs. numeric inputs.
# ==========================================================================

#%%
import pandas as pd

diabetes = pd.read_csv(r'/Users/mac/Desktop/NLP/Other Lectures/Explainability/Lecture Code/diabetes.csv')

print(diabetes.head())

#%%
# Load useful libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Separate Features and Target Variables
X = diabetes.drop(columns='Outcome')
y = diabetes['Outcome']

# Create Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, stratify =y, random_state = 42)

# Build the model
rf_clf = RandomForestClassifier(max_features=2, n_estimators =100 ,bootstrap = True)

rf_clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = rf_clf.predict(X_test)

# Classification Report
print(classification_report(y_pred, y_test))

#%%

import shap
import matplotlib.pyplot as plt

# load JS visualization code to notebook
shap.initjs()

# Create the explainer
explainer = shap.TreeExplainer(rf_clf)

shap_values = explainer.shap_values(X_test)
# %%
# ==========================================================================
# Step 1 - Load Dataset
# ==========================================================================
from sklearn.datasets import fetch_20newsgroups

full_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

class_names = []
for name in full_train.target_names:
    if 'misc' not in name:
        short_name = name.split('.')[-1]
    else:
        short_name = '.'.join(name.split('.')[-2:])
    class_names.append(short_name)
class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'

print(f"Class Names: {', '.join(class_names)}")

# %%
# ==========================================================================
# Step 2 - Necessary Packages
# ==========================================================================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

import shap
import numpy as np

# %%
# ==========================================================================
# Step 3 - Create a Smaller Training Subset & TF–IDF Vectors
# ==========================================================================
# Keep only 20% of the training set for speed
X_sub, _, y_sub, _ = train_test_split(
    full_train.data,
    full_train.target,
    test_size=0.8,
    random_state=42
)

vectorizer = TfidfVectorizer(lowercase=False, max_features=2000)
X_train_full = vectorizer.fit_transform(X_sub)  # numeric TF–IDF
y_train_full = y_sub

X_test_full = vectorizer.transform(newsgroups_test.data)
y_test_full = newsgroups_test.target

print(f"Reduced training shape: {X_train_full.shape}")
print(f"Test shape            : {X_test_full.shape}")

# %%
# ==========================================================================
# Step 4 - Train 3 Models
# ==========================================================================
print("\n[1] Logistic Regression")
model_lr = LogisticRegression(max_iter=200)
model_lr.fit(X_train_full, y_train_full)
pred_lr = model_lr.predict(X_test_full)
f1_lr = f1_score(y_test_full, pred_lr, average='weighted')
print(f"LR Weighted F1: {f1_lr:.3f}")

print("\n[2] Random Forest (n_estimators=50)")
model_rf = RandomForestClassifier(n_estimators=50, random_state=42)
model_rf.fit(X_train_full, y_train_full)
pred_rf = model_rf.predict(X_test_full)
f1_rf = f1_score(y_test_full, pred_rf, average='weighted')
print(f"RF Weighted F1: {f1_rf:.3f}")

print("\n[3] XGBoost (n_estimators=50)")
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_estimators=50)
model_xgb.fit(X_train_full, y_train_full)
pred_xgb = model_xgb.predict(X_test_full)
f1_xgb = f1_score(y_test_full, pred_xgb, average='weighted')
print(f"XGB Weighted F1: {f1_xgb:.3f}")

# %%
# ==========================================================================
# Step 5 - SHAP Explanation (Numeric)
#   We'll pick one test instance and explain with each model.
# ==========================================================================
test_idx = 1340
true_label = y_test_full[test_idx]

print(f"\n=== Explaining test instance #{test_idx} ===")
print("--------------------------------------------")
print("True Label:", class_names[true_label])

# The raw text, just for reference:
test_text = newsgroups_test.data[test_idx]
print("Text (truncated):", test_text[:200], "...")

# We'll define numeric predict_proba for each model
# -> input: numeric array, output: probability
def predict_proba_lr_numeric(x_num):
    return model_lr.predict_proba(x_num)

def predict_proba_rf_numeric(x_num):
    return model_rf.predict_proba(x_num)

def predict_proba_xgb_numeric(x_num):
    return model_xgb.predict_proba(x_num)

# Prepare the numeric row for the single test doc
test_vec = X_test_full[test_idx]  # still a sparse row
test_vec_dense = test_vec.toarray()  # shap likes dense format

# Let's build a background set from the training data (numeric)
bg_size = 50
num_train = X_train_full.shape[0]
rng = np.random.RandomState(0)
if num_train > bg_size:
    bg_indices = rng.choice(num_train, size=bg_size, replace=False)
else:
    bg_indices = np.arange(num_train)

background_data = X_train_full[bg_indices].toarray()

feature_names = vectorizer.get_feature_names_out()

def print_shap_top_features(shap_values, model_name, pred_label):
    """
    shap_values: list of arrays [class0 array, class1 array, ... classN array]
                 For a single instance, shap_values[c][0, feature_idx]
    pred_label: which class to explain
    model_name: string label for printing
    """
    class_shap = shap_values[pred_label][0]
    abs_order = np.argsort(np.abs(class_shap))[::-1]
    print(f"\n=== SHAP Explanation ({model_name}) ===")
    print(f"Predicted class = {class_names[pred_label]}")
    for i in abs_order[:6]:
        print(f"{feature_names[i]:<20} weight={class_shap[i]:.3f}")

# 1) Logistic Regression + SHAP
explainer_lr = shap.KernelExplainer(predict_proba_lr_numeric, background_data)
shap_vals_lr = explainer_lr.shap_values(test_vec_dense, nsamples=100)
pred_label_lr = model_lr.predict(test_vec)[0]
print_shap_top_features(shap_vals_lr, "Logistic Regression", pred_label_lr)

# 2) Random Forest + SHAP
explainer_rf = shap.KernelExplainer(predict_proba_rf_numeric, background_data)
shap_vals_rf = explainer_rf.shap_values(test_vec_dense, nsamples=100)
pred_label_rf = model_rf.predict(test_vec)[0]
print_shap_top_features(shap_vals_rf, "Random Forest", pred_label_rf)

# 3) XGBoost + SHAP
explainer_xgb = shap.KernelExplainer(predict_proba_xgb_numeric, background_data)
shap_vals_xgb = explainer_xgb.shap_values(test_vec_dense, nsamples=100)
pred_label_xgb = model_xgb.predict(test_vec)[0]
print_shap_top_features(shap_vals_xgb, "XGBoost", pred_label_xgb)

print("\n=== Done! ===")