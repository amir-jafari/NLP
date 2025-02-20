
#%%
import shap
import pandas as pd
import numpy as np

df = pd.read_csv('Customer Churn.csv')
df.head()
# Dropping all irrelevant columns
df.drop(columns=['customerID'], inplace = True)

df.dropna(inplace=True)

#%%

# Label Encoding features
categorical_feat =list(df.select_dtypes(include=["object"]))

# Using label encoder to transform string categories to integer labels
le = LabelEncoder()
for feat in categorical_feat:
    df[feat] = le.fit_transform(df[feat]).astype('int')