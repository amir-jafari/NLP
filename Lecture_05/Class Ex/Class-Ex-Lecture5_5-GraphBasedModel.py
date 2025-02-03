import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# Load the Iris dataset and split the dataset
# ============================================================
iris = load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================================
# Class_Ex1:
# Train a Decision Tree and measure accuracy on the test set.
# ------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')



print(20 * '-' + 'End Q1' + 20 * '-')

# ============================================================
# Class_Ex2:
# Adjust max_depth in Decision Tree to see if accuracy changes.
# Print accuracies for depths 1 to 5.
# ------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')



print(20 * '-' + 'End Q2' + 20 * '-')

# ============================================================
# Class_Ex3:
# Train a Random Forest and compare its accuracy to the Decision Tree.
# ------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')



print(20 * '-' + 'End Q3' + 20 * '-')

# ============================================================
# Class_Ex4:
# Train a Gradient Boosting Classifier and measure accuracy.
# ------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')



print(20 * '-' + 'End Q4' + 20 * '-')

# ============================================================
# Class_Ex5:
# Compare the accuracies of all three models on the Iris dataset.
# ------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')



print(20 * '-' + 'End Q5' + 20 * '-')
