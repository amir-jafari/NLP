# ===========================================================
# Gradient Boosting From Scratch with the Diabetes Dataset
# ===========================================================

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# 1 - Loading dataset
# -----------------------------------------------------------
data = load_diabetes()
X, y = data.data, data.target

# -----------------------------------------------------------
# 2 - Train-test split
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------------------------------------
# 3 - A minimal from-scratch gradient boosting
# -----------------------------------------------------------
def find_best_stump(X, residuals):
    n_samples, n_features = X.shape
    best_feature, best_threshold = None, None
    best_error = float('inf')
    best_left_value = 0.0
    best_right_value = 0.0

    for feature_idx in range(n_features):
        sorted_indices = np.argsort(X[:, feature_idx])
        X_sorted = X[sorted_indices, feature_idx]
        res_sorted = residuals[sorted_indices]

        for i in range(1, n_samples):
            threshold = (X_sorted[i - 1] + X_sorted[i]) / 2.0

            left_mask = (X[:, feature_idx] <= threshold)
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_value = np.mean(residuals[left_mask])
            right_value = np.mean(residuals[right_mask])

            left_errors = (residuals[left_mask] - left_value) ** 2
            right_errors = (residuals[right_mask] - right_value) ** 2
            total_error = np.sum(left_errors) + np.sum(right_errors)

            if total_error < best_error:
                best_error = total_error
                best_feature = feature_idx
                best_threshold = threshold
                best_left_value = left_value
                best_right_value = right_value

    return best_feature, best_threshold, best_left_value, best_right_value

class DecisionStump:
    def __init__(self, feature, threshold, left_value, right_value):
        self.feature = feature
        self.threshold = threshold
        self.left_value = left_value
        self.right_value = right_value

    def predict(self, X):
        return np.where(
            X[:, self.feature] <= self.threshold,
            self.left_value,
            self.right_value
        )

def gradient_boosting_regressor(X, y, n_estimators=5, learning_rate=0.1):
    """
    A minimal gradient boosting regressor using one-level decision stumps.
    """
    # Initialize predictions to the mean of y
    F = np.full_like(y, np.mean(y), dtype=float)
    stumps = []

    for _ in range(n_estimators):
        # 1. Compute residuals
        residuals = y - F

        # 2. Fit a stump to these residuals
        feat, thresh, left_val, right_val = find_best_stump(X, residuals)
        stump = DecisionStump(feat, thresh, left_val, right_val)

        # 3. Update predictions
        stump_pred = stump.predict(X)
        F += learning_rate * stump_pred

        stumps.append(stump)

    return stumps, F

# -----------------------------------------------------------
# 4 - Train and Use the model
# -----------------------------------------------------------
stumps, train_preds = gradient_boosting_regressor(
    X_train, y_train, n_estimators=5, learning_rate=0.1
)

train_mse = np.mean((y_train - train_preds)**2)

# Evaluate on the test set
F_test = np.full_like(y_test, np.mean(y_train), dtype=float)
for stump in stumps:
    stump_pred = stump.predict(X_test)
    F_test += 0.1 * stump_pred

test_mse = np.mean((y_test - F_test)**2)

print("From-scratch Gradient Boosting (Stumps) on Diabetes Dataset:")
print(f"Train MSE: {train_mse:.3f}")
print(f"Test  MSE: {test_mse:.3f}")
