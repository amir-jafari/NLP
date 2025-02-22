#%%---------------------------------------------------------------------------
import shap
import pandas as pd
import numpy as np
shap.initjs()
customer = pd.read_csv('Customer Churn.csv')
customer.head()

#%%---------------------------------------------------------------------------
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X = customer.drop("Churn", axis=1)
y = customer.Churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#%%---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_pred, y_test))
#%%---------------------------------------------------------------------------
explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test)
shap_values_list = [
    shap_values[..., 0],
    shap_values[..., 1],
]
shap.summary_plot(shap_values_list, X_test, plot_type="bar")

#%%---------------------------------------------------------------------------
shap_values_3d = explainer.shap_values(X_test)
shap_values_class1 = shap_values_3d[..., 1]
shap.summary_plot(shap_values_class1,
                  X_test,
                  feature_names=X_test.columns)

#%%---------------------------------------------------------------------------
shap_values_class0 = shap_values[..., 0]
shap_values_class1 = shap_values[..., 1]
shap.dependence_plot('Subscription  Length', shap_values_class0, X_test,interaction_index="Age")

#%%---------------------------------------------------------------------------
row_0_shap_values = shap_values_class0[0, :]
row_0_features = X_test.iloc[0, :]
shap.plots.force(
    explainer.expected_value[0],
    row_0_shap_values,
    row_0_features,
    matplotlib=True
)

#%%---------------------------------------------------------------------------
shap_values_class1 = shap_values[..., 1]
shap.decision_plot(
    base_value=explainer.expected_value[1],
    shap_values=shap_values_class1,
    features=X_test,
    feature_names=list(X_test.columns)
)
shap_values_class0 = shap_values[..., 0]
shap.decision_plot(
    base_value=explainer.expected_value[0],
    shap_values=shap_values_class0,
    features=X_test,
    # convert pd.Index -> list
    feature_names=list(X_test.columns)
)

#%%---------------------------------------------------------------------------
row_index = 0
shap_values_class1 = shap_values[..., 1]
explanation = shap.Explanation(
    values=shap_values_class1[row_index],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[row_index],
    feature_names=X_test.columns
)
shap.plots.waterfall(explanation)





