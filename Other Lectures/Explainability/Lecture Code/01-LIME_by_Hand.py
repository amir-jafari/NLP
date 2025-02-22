import numpy as np
from sklearn.linear_model import LinearRegression

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class LIMEHandExplainer:
    def __init__(self, model, kernel_width=0.75):
        self.model = model
        self.kernel_width = kernel_width
    def explain_instance(self, x1_0, x2_0, num_samples=10, noise_scale=0.5):
        x1_vals = []
        x2_vals = []
        f_vals  = []
        weights = []

        for _ in range(num_samples):
            x1_pert = x1_0 + np.random.normal(0, noise_scale)
            x2_pert = x2_0 + np.random.normal(0, noise_scale)
            y_pred = self.model(x1_pert, x2_pert)
            dist = np.sqrt((x1_pert - x1_0)**2 + (x2_pert - x2_0)**2)
            w = np.exp(-dist**2 / self.kernel_width)
            x1_vals.append(x1_pert)
            x2_vals.append(x2_pert)
            f_vals.append(y_pred)
            weights.append(w)

        X = np.column_stack([x1_vals, x2_vals])
        y = np.array(f_vals)
        w = np.array(weights)
        linreg = LinearRegression()
        linreg.fit(X, y, sample_weight=w)
        a0 = linreg.intercept_
        a1, a2 = linreg.coef_
        return a0, a1, a2, linreg

def my_model(x1, x2):
    z = 1.5 * x1 - 1.0 * x2
    return sigmoid(z)


if __name__ == "__main__":
    explainer = LIMEHandExplainer(model=my_model, kernel_width=0.75)
    x1_0, x2_0 = 2.0, 0.0
    y_pred = my_model(x1_0, x2_0)
    print(f"Target instance: (x1={x1_0}, x2={x2_0}), model output = {y_pred:.3f}")
    a0, a1, a2, linear_model = explainer.explain_instance(x1_0, x2_0,
                                                          num_samples=10,
                                                          noise_scale=0.5)
    print(f"\nLocal Surrogate Model (LIME):\n"
          f"   y_local = {a0:.3f} + {a1:.3f}*x1 + {a2:.3f}*x2")
    print("\nInterpretation:")
    print(f" - The coefficient for x1 is {a1:.3f}.")
    print(f" - The coefficient for x2 is {a2:.3f}.")
