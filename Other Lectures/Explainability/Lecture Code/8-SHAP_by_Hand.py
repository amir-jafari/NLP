# shap_by_hand.py
"""
A minimal example of SHAP by hand for a 2-feature model:
   f(x1, x2) = sigmoid(1.5*x1 - 1.0*x2).
We compute Shapley values at (2,0) using a baseline of (0,0).
"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Our model
def model(x1, x2):
    return sigmoid(1.5 * x1 - 1.0 * x2)

# -------------------------------------------------------------------------
# 1) Define instance to explain and a baseline
# -------------------------------------------------------------------------
x1_0, x2_0 = 2.0, 0.0
# baseline = (0,0)

f_instance = model(x1_0, x2_0)
f_baseline = model(0, 0)
print(f"Instance: (2,0) => f={f_instance:.3f}")
print(f"Baseline: (0,0) => f={f_baseline:.3f}")

# -------------------------------------------------------------------------
# 2) Enumerate subsets for 2-feature SHAP
#    S can be {}, {x1}, {x2}, {x1,x2}.
# -------------------------------------------------------------------------

# f({}): both features at baseline => (0,0)
f_empty = model(0,0)

# f({x1}): x1=2 (actual), x2=0 (baseline)
f_x1 = model(2,0)

# f({x2}): x2=0 (actual) but x1=0 (baseline) => still (0,0) in this contrived example
f_x2 = model(0,0)

# f({x1,x2}): x1=2, x2=0 => actual
f_x1x2 = model(2,0)

# -------------------------------------------------------------------------
# 3) Shapley formula
# -------------------------------------------------------------------------
# phi_x1 = 1/2 * [ (f({x1}) - f({})) + (f({x1,x2}) - f({x2})) ]
phi_x1 = 0.5 * ((f_x1 - f_empty) + (f_x1x2 - f_x2))

# phi_x2 = 1/2 * [ (f({x2}) - f({})) + (f({x1,x2}) - f({x1})) ]
phi_x2 = 0.5 * ((f_x2 - f_empty) + (f_x1x2 - f_x1))

print(f"Shapley values:")
print(f"  phi_x1 = {phi_x1:.3f}")
print(f"  phi_x2 = {phi_x2:.3f}")

# -------------------------------------------------------------------------
# 4) Check that baseline + sum of SHAP = final
# -------------------------------------------------------------------------
pred_sum_check = f_empty + phi_x1 + phi_x2
print(f"\nCheck => baseline + phi_x1 + phi_x2 = {pred_sum_check:.3f}")
print(f"Final model prediction             = {f_x1x2:.3f}")
