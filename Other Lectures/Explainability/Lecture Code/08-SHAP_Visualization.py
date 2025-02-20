# =====================================================================
# 'Customer Churn' dataset is a Perfect dataset for Predictive
# Prescription analysis.
#
# url: https://www.kaggle.com/datasets/royjafari/customer-churn
# =====================================================================

#%%
import shap
import pandas as pd
import numpy as np
shap.initjs()

customer = pd.read_csv("data/customer_churn.csv")
customer.head()
