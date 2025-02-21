# =====================================================================
# 'Customer Churn' dataset is a Perfect dataset for Predictive
# Prescription analysis.
#
# url: https://www.kaggle.com/datasets/royjafari/customer-churn
# =====================================================================
#%%
import pandas as pd
import numpy as np

# For model and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# For LIME explanations
import lime
from lime import lime_tabular
from lime import submodular_pick

# For opening the HTML files automatically in the browser
import webbrowser
import os

customer = pd.read_csv('Customer Churn.csv')

print(customer.head())

# Drop irrelevant columns
if 'customerID' in customer.columns:
    customer.drop(columns=['customerID'], inplace=True)

# Drop missing values
customer.dropna(inplace=True)

#%%
# =====================================================================
# Identify categorical columns and label-encode them
# =====================================================================
categorical_feat = list(customer.select_dtypes(include=["object"]))
le = LabelEncoder()
for feat in categorical_feat:
    customer[feat] = le.fit_transform(customer[feat])

print("\nData after preprocessing:")
print(customer.head())

#%%
# =====================================================================
# Model Training and Evaluation
# =====================================================================
features = customer.drop(columns=['Churn'])
labels = customer['Churn']

x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=123
)

model = XGBClassifier(n_estimators=300, random_state=123)
model.fit(x_train, y_train)

#%%
# =====================================================================
# Explainer using LIME
# =====================================================================
np.random.seed(123)
predict_fn = lambda x: model.predict_proba(x)
# Defining the LIME explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(customer[features.columns].astype(int).values,
                                                   mode='classification',
                                                   class_names=['Did not Churn', 'Churn'],
                                                   training_labels=customer['Churn'],
                                                   feature_names=features.columns)
# using LIME to get the explanations
i = 5
exp=explainer.explain_instance(customer.loc[i,features.columns].astype(int).values, predict_fn, num_features=5)

#%%
# =====================================================================
# Local Interpretability
# =====================================================================

local_html_file = "local_explanation.html"
exp.save_to_file(local_html_file)
print(f"Local explanation saved as {local_html_file}")

# Automatically open the local explanation in the default browser
webbrowser.open_new_tab(os.path.abspath(local_html_file))

#%%
# =====================================================================
# Global SP-LIME Interpretability
# =====================================================================
sp_exp = submodular_pick.SubmodularPick(
    explainer,
    data=features.values,
    predict_fn=predict_fn,
    sample_size=100,     # how many data points to sample
    num_exps_desired=5,  # how many local explanations to generate
    num_features=5
)

for idx, ex in enumerate(sp_exp.sp_explanations):
    html_name = f"sp_explanation_{idx+1}.html"
    ex.save_to_file(html_name)
    print(f"SP-LIME explanation #{idx+1} saved as {html_name}")

    # Open each SP-LIME explanation automatically
    webbrowser.open_new_tab(os.path.abspath(html_name))

#%%
# =====================================================================
# Local explanation for class versicolor
# =====================================================================
import matplotlib.pyplot as plt

local_exp = exp.as_list()
# Separate feature names and their contribution weights
labels, weights = zip(*local_exp)

plt.barh(labels, weights, color='green')
plt.xlabel('Feature Contribution')
plt.title('LIME Explanation for One Prediction')
plt.gca().invert_yaxis()  # Highest contribution at top

plt.tight_layout()
plt.show()