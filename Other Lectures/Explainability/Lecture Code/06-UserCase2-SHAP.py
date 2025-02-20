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

# Just for neatness in printing
class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'

print(f"Class Names: {', '.join(class_names)}")

# %%
# ==========================================================================
# Step 2 - Necessary Packages
# ==========================================================================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
import numpy as np

# Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import f1_score

# %%
# ==========================================================================
# Step 3 - Create a Smaller Training Subset & TF–IDF Vectors
# ==========================================================================

# =====================================================
# The below code is to reduce the dataset to only 20%
# We only keep 20% of the whole dataset
# =====================================================
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

print("\n[2] Random Forest with n_estimators=50")
model_rf = RandomForestClassifier(n_estimators=50, random_state=42)
model_rf.fit(X_train_full, y_train_full)
pred_rf = model_rf.predict(X_test_full)
f1_rf = f1_score(y_test_full, pred_rf, average='weighted')
print(f"RF Weighted F1: {f1_rf:.3f}")

print("\n[3] XGBoost with n_estimators=50")
model_xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_estimators=50
)
model_xgb.fit(X_train_full, y_train_full)
pred_xgb = model_xgb.predict(X_test_full)
f1_xgb = f1_score(y_test_full, pred_xgb, average='weighted')
print(f"XGB Weighted F1: {f1_xgb:.3f}")

# %%
# ==========================================================================
# Step 5 - SHAP Explanation (KernelExplainer)
# ==========================================================================

test_idx = 0
test_text = newsgroups_test.data[test_idx]
true_label = newsgroups_test.target[test_idx]


background_size = 100

indices = np.random.choice(X_train_full.shape[0], background_size, replace=False)
background_data = X_train_full[indices].toarray()

# We want to get feature names for easier debugging
feature_names = vectorizer.get_feature_names_out()

# Function to print out top contributing features
def print_top_features(shap_values, k=10):
    """
    shap_values: array of shape (n_features,) with the SHAP values.
    k: number of top features to print
    """
    abs_vals = np.abs(shap_values)
    sorted_indices = np.argsort(abs_vals)[::-1]
    top_indices = sorted_indices[:k]
    for idx in top_indices:
        # feature_names[idx] => name, shap_values[idx] => value
        print(f"{feature_names[idx]:<20} => SHAP value = {shap_values[idx]:.3f}")

# ---------------------
# Step 5.1 - LR Explanation
# ---------------------
def lr_predict_proba(X):
    return model_lr.predict_proba(X)

print("\n--- SHAP for Logistic Regression ---")
explainer_lr = shap.KernelExplainer(lr_predict_proba, background_data)

# X_test_full[[test_idx]] is a slice of shape (1, n_features)
shap_values_lr = explainer_lr.shap_values(X_test_full[[test_idx]].toarray())

# shap_values_lr is a list (one array per class), shape => (#classes, 1, n_features)
predicted_class_lr = pred_lr[test_idx]
# Extract the SHAP values for the predicted class => shape (1, n_features)
shap_values_class_lr = shap_values_lr[predicted_class_lr][0]
print_top_features(shap_values_class_lr, k=10)

# ---------------------
# Step 5.2 - RF Explanation
# ---------------------
def rf_predict_proba(X):
    return model_rf.predict_proba(X)

print("\n--- SHAP for Random Forest ---")
explainer_rf = shap.KernelExplainer(rf_predict_proba, background_data)
shap_values_rf = explainer_rf.shap_values(X_test_full[[test_idx]].toarray())

predicted_class_rf = pred_rf[test_idx]
shap_values_class_rf = shap_values_rf[predicted_class_rf][0]
print_top_features(shap_values_class_rf, k=10)

# ---------------------
# Step 5.3 - XGBoost Explanation
# ---------------------
def xgb_predict_proba(X):
    return model_xgb.predict_proba(X)

print("\n--- SHAP for XGBoost ---")
explainer_xgb = shap.KernelExplainer(xgb_predict_proba, background_data)
shap_values_xgb = explainer_xgb.shap_values(X_test_full[[test_idx]].toarray())

predicted_class_xgb = pred_xgb[test_idx]
shap_values_class_xgb = shap_values_xgb[predicted_class_xgb][0]
print_top_features(shap_values_class_xgb, k=10)

