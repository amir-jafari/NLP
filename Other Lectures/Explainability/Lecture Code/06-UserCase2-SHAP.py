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


