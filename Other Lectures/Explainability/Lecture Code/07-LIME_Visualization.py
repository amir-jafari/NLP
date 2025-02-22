#%%---------------------------------------------------------------------------
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

if 'customerID' in customer.columns:
    customer.drop(columns=['customerID'], inplace=True)
customer.dropna(inplace=True)

#%%---------------------------------------------------------------------------
categorical_feat = list(customer.select_dtypes(include=["object"]))
le = LabelEncoder()
for feat in categorical_feat:
    customer[feat] = le.fit_transform(customer[feat])
print("\nData after preprocessing:")
print(customer.head())

#%%---------------------------------------------------------------------------
features = customer.drop(columns=['Churn'])
labels = customer['Churn']
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=123
)
model = XGBClassifier(n_estimators=300, random_state=123)
model.fit(x_train, y_train)

#%%---------------------------------------------------------------------------
np.random.seed(123)
predict_fn = lambda x: model.predict_proba(x)
explainer = lime.lime_tabular.LimeTabularExplainer(customer[features.columns].astype(int).values,
                                                   mode='classification',
                                                   class_names=['Did not Churn', 'Churn'],
                                                   training_labels=customer['Churn'],
                                                   feature_names=features.columns)
i = 5
exp=explainer.explain_instance(customer.loc[i,features.columns].astype(int).values, predict_fn, num_features=5)

#%%---------------------------------------------------------------------------
local_html_file = "local_explanation.html"
exp.save_to_file(local_html_file)
print(f"Local explanation saved as {local_html_file}")
webbrowser.open_new_tab(os.path.abspath(local_html_file))

#%%---------------------------------------------------------------------------
sp_exp = submodular_pick.SubmodularPick(
    explainer,
    data=features.values,
    predict_fn=predict_fn,
    sample_size=100,
    num_exps_desired=5,
    num_features=5
)

for idx, ex in enumerate(sp_exp.sp_explanations):
    html_name = f"sp_explanation_{idx+1}.html"
    ex.save_to_file(html_name)
    print(f"SP-LIME explanation #{idx+1} saved as {html_name}")
    webbrowser.open_new_tab(os.path.abspath(html_name))

#%%---------------------------------------------------------------------------
import matplotlib.pyplot as plt

local_exp = exp.as_list()
labels, weights = zip(*local_exp)

plt.barh(labels, weights, color='green')
plt.xlabel('Feature Contribution')
plt.title('LIME Explanation for One Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()