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
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Separate Features and Target Variables
X = diabetes.drop(columns='Outcome')
y = diabetes['Outcome']

# Create Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, stratify =y, random_state = 42)

# Build the Random Forest model
rf_clf = RandomForestClassifier(max_features=2, n_estimators =100 ,bootstrap = True)

# Build the Logistic Regression Model
lr_clf = LogisticRegression(max_iter=1000)

# Build the XGBoost Model
xgb_clf = XGBClassifier(n_estimators=100,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss')

# Fit the model on training data
lr_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred_rf = rf_clf.predict(X_test)
y_pred_lr = lr_clf.predict(X_test)
y_pred_xgb = xgb_clf.predict(X_test)


#%%

import shap
import matplotlib.pyplot as plt

# load JS visualization code to notebook
shap.initjs()

# Create the explainer
explainer_rf = shap.TreeExplainer(rf_clf)
shap_values_rf = explainer_rf.shap_values(X_test)
print("SHAP values for Random Forest:")
print(shap_values_rf)

explainer_xgb = shap.TreeExplainer(xgb_clf)
shap_values_xgb = explainer_xgb.shap_values(X_test)
print("\nSHAP values for XGBoost:")
print(shap_values_xgb)

# For Logistic Regression (linear model), use LinearExplainer or KernelExplainer
explainer_lr = shap.LinearExplainer(lr_clf, X_train)
shap_values_lr = explainer_lr.shap_values(X_test)
print("\nSHAP values for Logistic Regression:")
print(shap_values_lr)
