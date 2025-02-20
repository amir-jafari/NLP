# -*- coding: utf-8 -*-
"""
20 Newsgroups Classification with SHAP (Model-Specific)
Logistic Regression => shap.LinearExplainer (multinomial)
Random Forest, XGBoost => shap.TreeExplainer
"""

# =============================================================================
# Step 1 - Load Dataset
# =============================================================================
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

# =============================================================================
# Step 2 - Import Packages
# =============================================================================
import shap
import numpy as np
import scipy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# =============================================================================
# Step 3 - Create a Smaller Training Subset & TF–IDF Vectors
# =============================================================================
X_sub, _, y_sub, _ = train_test_split(
    full_train.data,
    full_train.target,
    test_size=0.8,
    random_state=42
)

vectorizer = TfidfVectorizer(lowercase=False, max_features=2000)
X_train_full = vectorizer.fit_transform(X_sub)
y_train_full = y_sub

X_test_full = vectorizer.transform(newsgroups_test.data)
y_test_full = newsgroups_test.target

# =============================================================================
# Step 4 - Train & Evaluate Three Models
# =============================================================================
print("\n[1] Logistic Regression (multinomial)")
model_lr = LogisticRegression(max_iter=200, solver='lbfgs')
model_lr.fit(X_train_full, y_train_full)
pred_lr = model_lr.predict(X_test_full)
print(f"LR Weighted F1: {f1_score(y_test_full, pred_lr, average='weighted'):.3f}")

print("\n[2] Random Forest (n_estimators=50)")
model_rf = RandomForestClassifier(n_estimators=50, random_state=42)
model_rf.fit(X_train_full, y_train_full)
pred_rf = model_rf.predict(X_test_full)
print(f"RF Weighted F1: {f1_score(y_test_full, pred_rf, average='weighted'):.3f}")

print("\n[3] XGBoost (n_estimators=50)")
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_estimators=50)
model_xgb.fit(X_train_full, y_train_full)
pred_xgb = model_xgb.predict(X_test_full)
print(f"XGB Weighted F1: {f1_score(y_test_full, pred_xgb, average='weighted'):.3f}")

# =============================================================================
# Step 5 - SHAP Explanation of a Single Test Instance
# =============================================================================
test_idx = 0
test_text = newsgroups_test.data[test_idx]
true_label = y_test_full[test_idx]

print(f"\n=== Explaining Test Instance #{test_idx} ===")
print("True Label:", class_names[true_label])
print("Text (truncated):", test_text[:200], "...")

feature_names = vectorizer.get_feature_names_out()

def print_top_features(shap_values, top_k=10):
    abs_vals = np.abs(shap_values)
    sorted_indices = np.argsort(abs_vals)[::-1][:top_k]
    for idx in sorted_indices:
        print(f"{feature_names[idx]:<20} => SHAP = {shap_values[idx]:.3f}")

# --- SHAP for Logistic Regression ---
X_train_lr = X_train_full.toarray() if scipy.sparse.issparse(X_train_full) else X_train_full
X_test_lr  = X_test_full.toarray()  if scipy.sparse.issparse(X_test_full)  else X_test_full

explainer_lr = shap.LinearExplainer(model_lr, X_train_lr)
shap_ex_lr = explainer_lr(X_test_lr[test_idx:test_idx+1])
pred_cl_lr = pred_lr[test_idx]
shap_vals_lr = shap_ex_lr.values[0, pred_cl_lr, :]

print("\n[Logistic Regression]")
print(f"Predicted class: {class_names[pred_cl_lr]}")
print_top_features(shap_vals_lr, top_k=10)

# --- SHAP for Random Forest ---
X_train_rf = X_train_full.toarray()
X_test_rf  = X_test_full.toarray()

explainer_rf = shap.TreeExplainer(
    model_rf, data=X_train_rf,
    feature_perturbation='interventional'
)
shap_ex_rf = explainer_rf(X_test_rf[test_idx:test_idx+1], check_additivity=False)
pred_cl_rf = pred_rf[test_idx]
shap_vals_rf = shap_ex_rf.values[0, pred_cl_rf, :]

print("\n[Random Forest]")
print(f"Predicted class: {class_names[pred_cl_rf]}")
print_top_features(shap_vals_rf, top_k=10)

# --- SHAP for XGBoost ---
X_train_xgb = X_train_full.toarray()
X_test_xgb  = X_test_full.toarray()

explainer_xgb = shap.TreeExplainer(
    model_xgb, data=X_train_xgb,
    feature_perturbation='interventional'
)
shap_ex_xgb = explainer_xgb(X_test_xgb[test_idx:test_idx+1], check_additivity=False)
pred_cl_xgb = pred_xgb[test_idx]
shap_vals_xgb = shap_ex_xgb.values[0, pred_cl_xgb, :]

print("\n[XGBoost]")
print(f"Predicted class: {class_names[pred_cl_xgb]}")
print_top_features(shap_vals_xgb, top_k=10)
