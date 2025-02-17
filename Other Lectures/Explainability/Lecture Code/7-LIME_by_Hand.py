# lime_by_hand.py
"""
A minimal example of LIME by hand for a tiny 2-feature model:
   f(x1, x2) = sigmoid(1.5*x1 - 1.0*x2).
We locally approximate f around a chosen instance (2,0) by a linear surrogate.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# "Black-box" model
def model(x1, x2):
    return sigmoid(1.5 * x1 - 1.0 * x2)

# -------------------------------------------------------------------------
# 1) Target instance to explain:
# -------------------------------------------------------------------------
x1_0, x2_0 = 2.0, 0.0
f0 = model(x1_0, x2_0)
print(f"Target instance: (x1=2, x2=0), model output = {f0:.3f}")

# -------------------------------------------------------------------------
# 2) Generate local perturbations around (2,0):
# -------------------------------------------------------------------------
N = 10
x1_vals = []
x2_vals = []
f_vals  = []
weights = []

# sample around (2,0) using a small Gaussian
for _ in range(N):
    x1_pert = x1_0 + np.random.normal(0, 0.5)
    x2_pert = x2_0 + np.random.normal(0, 0.5)

    # Model prediction for perturbed sample
    y_pred = model(x1_pert, x2_pert)

    # Distance-based weighting:
    dist = np.sqrt((x1_pert - x1_0)**2 + (x2_pert - x2_0)**2)
    w = np.exp(-dist**2 / 0.75)

    x1_vals.append(x1_pert)
    x2_vals.append(x2_pert)
    f_vals.append(y_pred)
    weights.append(w)

X = np.column_stack([x1_vals, x2_vals])
y = np.array(f_vals)
w = np.array(weights)

# -------------------------------------------------------------------------
# 3) Fit a weighted linear surrogate
# -------------------------------------------------------------------------
linreg = LinearRegression()
linreg.fit(X, y, sample_weight=w)

a0 = linreg.intercept_
a1, a2 = linreg.coef_

print("Local Surrogate Model (LIME):")
print(f"   y_local = {a0:.3f} + {a1:.3f}*x1 + {a2:.3f}*x2")

# -------------------------------------------------------------------------
# 4) Interpretation:
# -------------------------------------------------------------------------
print("\nInterpretation:")
print(f" - The coefficient for x1 is {a1:.3f} (in the local region).")
print(f" - The coefficient for x2 is {a2:.3f}.")
print("Higher magnitude => more importance for that feature locally.")
