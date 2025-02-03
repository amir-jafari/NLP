# -------------------------------------------------------------
# Gradient Boosting using scikit-learn
# -------------------------------------------------------------

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


data = load_diabetes()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Gradient Boosting with scikit-learn on Diabetes dataset:")
print(f"Train MSE: {train_mse:.3f}")
print(f"Test  MSE: {test_mse:.3f}")
