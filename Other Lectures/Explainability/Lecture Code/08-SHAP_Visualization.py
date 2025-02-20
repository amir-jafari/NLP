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

customer = pd.read_csv('Customer Churn.csv')
customer.head()

#%%
# =======================================================================
# Model Training and Evaluation
#   -1. Create X and y using a target column and split the dataset into train and test.
#   -2.Train Random Forest Classifier on the training set.
#   -3. Make predictions using a testing set.
#   -4. Display classification report.
# =======================================================================
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X = customer.drop("Churn", axis=1)
y = customer.Churn

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train a machine learning model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = clf.predict(X_test)

# Classification Report
print(classification_report(y_pred, y_test))



#%%
# =====================================================================
# Graph 1 - Summary Plot: Side‐by‐Side Bar Chart for Class0 and Class1
# the two‐color bar chart (one column for Class0 in blue, one for Class1
# in red, per feature), you need to give summary_plot a list of the two
# (n_samples, n_features) arrays—one for Class0, one for Class1.
#
# =====================================================================
explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test)

shap_values_list = [
    shap_values[..., 0],  # class 0
    shap_values[..., 1],  # class 1
]

shap.summary_plot(shap_values_list, X_test, plot_type="bar")

#%%
# =====================================================================
# Graph 2 - Summary Plot: 3D Side‐by‐Side Bar Chart for Class0 and Class1
# =====================================================================
shap_values_3d = explainer.shap_values(X_test)

shap_values_class1 = shap_values_3d[..., 1]

shap.summary_plot(shap_values_class1,
                  X_test,
                  feature_names=X_test.columns)

#%%
# =====================================================================
# Graph 3 - Dependence Plot
# =====================================================================
# shap_values is shape (945, 15, 2)
shap_values_class0 = shap_values[..., 0]  # shape (945, 15)
shap_values_class1 = shap_values[..., 1]  # shape (945, 15)

shap.dependence_plot('Subscription  Length', shap_values_class0, X_test,interaction_index="Age")

#%%
# =====================================================================
# Graph 4 - Force Plot
# =====================================================================
row_0_shap_values = shap_values_class0[0, :]
row_0_features = X_test.iloc[0, :]

shap.plots.force(
    explainer.expected_value[0],
    row_0_shap_values,
    row_0_features,
    matplotlib=True
)

#%%
# =====================================================================
# Graph 5 - Decision Plot
# =====================================================================
shap_values_class1 = shap_values[..., 1]

shap.decision_plot(
    base_value=explainer.expected_value[1],
    shap_values=shap_values_class1,
    features=X_test,
    # convert pd.Index -> list
    feature_names=list(X_test.columns),
)

shap_values_class0 = shap_values[..., 0]

shap.decision_plot(
    base_value=explainer.expected_value[0],
    shap_values=shap_values_class0,
    features=X_test,
    # convert pd.Index -> list
    feature_names=list(X_test.columns),
)









