import numpy as np
from sklearn.linear_model import LinearRegression

# Sigmoid function we are using for the model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# The model we want to explain (it's a simple function)
def model(x1, x2):
    return sigmoid(1.5 * x1 - 1.0 * x2)

# 1) Pick a target instance we want to explain
x1_0, x2_0 = 2.0, 0.0
f0 = model(x1_0, x2_0)
print(f"Target instance: (x1=2, x2=0), model output = {f0:.3f}")

# 2) Now, let's make some small changes (perturbations) to the instance (x1, x2)
N = 10  # Number of samples we're going to create around (2, 0)
x1_vals = []
x2_vals = []
f_vals  = []
weights = []

# Create small random changes (Gaussian noise) around our target
for _ in range(N):
    x1_pert = x1_0 + np.random.normal(0, 0.5)  # Perturb x1 by adding random noise
    x2_pert = x2_0 + np.random.normal(0, 0.5)  # Perturb x2 similarly

    # Get the model's prediction for the perturbed point
    y_pred = model(x1_pert, x2_pert)

    # Assign a weight based on how close the perturbed point is to the target (distance-based weighting)
    dist = np.sqrt((x1_pert - x1_0)**2 + (x2_pert - x2_0)**2)  # Euclidean distance
    w = np.exp(-dist**2 / 0.75)  # Closer points should have higher weight

    x1_vals.append(x1_pert)
    x2_vals.append(x2_pert)
    f_vals.append(y_pred)
    weights.append(w)

# 3) Now we fit a linear model (surrogate) to these perturbed points
X = np.column_stack([x1_vals, x2_vals])  # Features (perturbed x1 and x2)
y = np.array(f_vals)  # Model predictions for these perturbed points
w = np.array(weights)  # Weights based on distance

# Fit a linear model using the weighted points
linreg = LinearRegression()
linreg.fit(X, y, sample_weight=w)

a0 = linreg.intercept_
a1, a2 = linreg.coef_

print("Local Surrogate Model (LIME):")
print(f"   y_local = {a0:.3f} + {a1:.3f}*x1 + {a2:.3f}*x2")

# 4) Now let's interpret the coefficients
print("\nInterpretation:")
print(f" - The coefficient for x1 is {a1:.3f}.")
print(f" - The coefficient for x2 is {a2:.3f}.")
print("The larger the coefficient, the more important that feature is locally.")
