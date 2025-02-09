# ---------------------------------------------------------------------
# Build Classifier
# ---------------------------------------------------------------------

#%%
# Load useful libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

diabetes_data = pd.read_csv('diabetes.csv')

# Separate Features and Target Variables
X = diabetes_data.drop(columns='Outcome')
y = diabetes_data['Outcome']

# Create Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,
                                                	stratify =y,
                                                	random_state = 13)

# Build the model
rf_clf = RandomForestClassifier(max_features=2, n_estimators =100 ,bootstrap = True)

rf_clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = rf_clf.predict(X_test)

# Classification Report
print(classification_report(y_pred, y_test))


#%%
# ---------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# If you did not install the library 'LIME', Please use the below command in the terminal (shell)
# pip install lime
# pip3 install lime
# conda install lime
# ---------------------------------------------------------------------

# Import the LimeTabularExplainer module
from lime.lime_tabular import LimeTabularExplainer

# Get the class names
class_names = ['Has diabetes', 'No diabetes']

# Get the feature names
feature_names = list(X_train.columns)

# Fit the Explainer on the training data set using the LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values, feature_names =
                                 feature_names,
                                 class_names = class_names,
                                 mode = 'classification')

#%%
# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# If you did not install the library 'SHAP', Please use the below command in the terminal (shell)
# pip install shap
# pip3 install shap
# conda install shap
# ---------------------------------------------------------------------


import shap
import matplotlib.pyplot as plt

# load JS visualization code
shap.initjs()

# Create the explainer
explainer = shap.TreeExplainer(rf_clf)

shap_values = explainer.shap_values(X_test)

# Variable Importance with Summary Plot
print("Variable Importance Plot - Global Interpretation")
figure = plt.figure()
shap.summary_plot(shap_values, X_test)

# Summary Plot on a Specific Label
shap.summary_plot(shap_values[1], X_test)