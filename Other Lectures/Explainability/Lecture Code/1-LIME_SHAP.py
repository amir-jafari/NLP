# ---------------------------------------------------------------------
# 1. DATASET & MODEL
# ---------------------------------------------------------------------
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='Outcome')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=13
)

rf_clf = RandomForestClassifier(n_estimators=50, max_features=2, bootstrap=True, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------
# 2. LIME
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# If you did not install the library 'LIME', Please use the below command in the terminal (shell)
# pip install lime
# pip3 install lime
# conda install lime
# ---------------------------------------------------------------------

from lime.lime_tabular import LimeTabularExplainer

class_names = ['Class 0', 'Class 1']
feature_names = X_train.columns.tolist()
X_train_small = X_train.sample(n=100, random_state=42)
explainer = LimeTabularExplainer(
    training_data=X_train_small.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

instance_index = 0
one_row_df = X_test.iloc[[instance_index]]
explanation = explainer.explain_instance(
    data_row=one_row_df.values[0],
    predict_fn=rf_clf.predict_proba,
    num_features=5
)

print("LIME Explanation for instance", instance_index)
for feat, weight in explanation.as_list():
    print(feat, weight)

explanation.show_in_notebook(show_table=True)

# ---------------------------------------------------------------------
# 3. SHAP (use smaller subset of X_test)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# If you did not install the library 'SHAP', Please use the below command in the terminal (shell)
# pip install shap
# pip3 install shap
# conda install shap
# ---------------------------------------------------------------------

import shap
shap.initjs()
explainer_shap = shap.TreeExplainer(rf_clf)
X_test_sample = X_test.sample(n=50, random_state=42)
shap_values = explainer_shap.shap_values(X_test_sample)

if isinstance(shap_values, list):
    shap.summary_plot(shap_values[0], X_test_sample)
    shap.summary_plot(shap_values[1], X_test_sample)
else:
    shap.summary_plot(shap_values, X_test_sample)
