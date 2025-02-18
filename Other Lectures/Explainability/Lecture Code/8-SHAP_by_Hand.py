import numpy as np

# Sigmoid function, this is our "black-box" model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# The model we're explaining: f(x1, x2) = sigmoid(1.5*x1 - 1.0*x2)
def model(x1, x2):
    return sigmoid(1.5 * x1 - 1.0 * x2)

# -------------------------------------------------------------------------
# 1) Define the instance to explain and the baseline
# -------------------------------------------------------------------------
# Target instance is (2, 0), and the baseline is (0, 0)
x1_0, x2_0 = 2.0, 0.0
f_instance = model(x1_0, x2_0)  # The model's prediction for our instance
f_baseline = model(0, 0)  # The baseline prediction (model output at (0,0))

print(f"Instance: (2,0) => f={f_instance:.3f}")
print(f"Baseline: (0,0) => f={f_baseline:.3f}")

# -------------------------------------------------------------------------
# 2) Enumerate subsets for 2-feature SHAP
# -------------------------------------------------------------------------
# For SHAP, we calculate the model output for different subsets of features.

# f({}): Both features at baseline => (0,0)
f_empty = model(0, 0)

# f({x1}): Only x1 = 2 (actual), x2 = 0 (baseline)
f_x1 = model(2, 0)

# f({x2}): Only x2 = 0 (actual), x1 = 0 (baseline) => still (0,0) in this example
f_x2 = model(0, 0)

# f({x1, x2}): Both x1 = 2, x2 = 0 => this is our target instance
f_x1x2 = model(2, 0)

# -------------------------------------------------------------------------
# 3) Calculate Shapley values using the formula
# -------------------------------------------------------------------------
# Now we compute the Shapley values for each feature. This is the core idea behind SHAP:
# We evaluate how much each feature contributes to the model prediction, considering all subsets.

# phi_x1 = 1/2 * [ (f({x1}) - f({})) + (f({x1,x2}) - f({x2})) ]
phi_x1 = 0.5 * ((f_x1 - f_empty) + (f_x1x2 - f_x2))

# phi_x2 = 1/2 * [ (f({x2}) - f({})) + (f({x1,x2}) - f({x1})) ]
phi_x2 = 0.5 * ((f_x2 - f_empty) + (f_x1x2 - f_x1))

print(f"Shapley values:")
print(f"  phi_x1 = {phi_x1:.3f}")
print(f"  phi_x2 = {phi_x2:.3f}")

# -------------------------------------------------------------------------
# 4) Check if the Shapley values sum up to the final model output
# -------------------------------------------------------------------------
# The sum of the baseline and the Shapley values should give us the model's output for the instance.

pred_sum_check = f_empty + phi_x1 + phi_x2
print(f"\nCheck => baseline + phi_x1 + phi_x2 = {pred_sum_check:.3f}")
print(f"Final model prediction             = {f_x1x2:.3f}")
