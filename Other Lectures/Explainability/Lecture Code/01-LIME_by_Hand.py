import numpy as np
from sklearn.linear_model import LinearRegression

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class LIMEHandExplainer:
    """
    A simple LIME-like explainer for binary models of the form model(x1, x2).
    """

    def __init__(self, model, kernel_width=0.75):
        """
        :param model: A function that takes (x1, x2) and returns a probability (0-1).
        :param kernel_width: Determines how quickly weights decay with distance.
        """
        self.model = model
        self.kernel_width = kernel_width

    def explain_instance(self, x1_0, x2_0, num_samples=10, noise_scale=0.5):
        """
        Explains a target instance (x1_0, x2_0) by:
          1) Generating small perturbations around the instance
          2) Weighting them by distance
          3) Fitting a local linear surrogate model
          4) Returning the surrogate's intercept & coefficients

        :param x1_0: float, the x1 value of the target instance
        :param x2_0: float, the x2 value of the target instance
        :param num_samples: how many perturbed samples to generate
        :param noise_scale: std dev for the random noise around x1_0, x2_0
        :return: (intercept, coeff_x1, coeff_x2, fitted_model)
        """
        # 1) Generate perturbations
        x1_vals = []
        x2_vals = []
        f_vals  = []
        weights = []

        for _ in range(num_samples):
            x1_pert = x1_0 + np.random.normal(0, noise_scale)
            x2_pert = x2_0 + np.random.normal(0, noise_scale)

            # Model prediction for this perturbed sample
            y_pred = self.model(x1_pert, x2_pert)

            # Distance-based weighting (Gaussian kernel)
            dist = np.sqrt((x1_pert - x1_0)**2 + (x2_pert - x2_0)**2)
            w = np.exp(-dist**2 / self.kernel_width)

            x1_vals.append(x1_pert)
            x2_vals.append(x2_pert)
            f_vals.append(y_pred)
            weights.append(w)

        # 2) Fit local linear surrogate
        X = np.column_stack([x1_vals, x2_vals])
        y = np.array(f_vals)
        w = np.array(weights)

        linreg = LinearRegression()
        linreg.fit(X, y, sample_weight=w)

        a0 = linreg.intercept_
        a1, a2 = linreg.coef_

        return a0, a1, a2, linreg

def my_model(x1, x2):
    # This is the same function you had: model(x1, x2) = sigmoid(1.5*x1 - 1.0*x2)
    z = 1.5 * x1 - 1.0 * x2
    return sigmoid(z)


if __name__ == "__main__":
    # 1) Instantiate the explainer
    explainer = LIMEHandExplainer(model=my_model, kernel_width=0.75)

    # 2) Pick an instance to explain
    x1_0, x2_0 = 2.0, 0.0
    y_pred = my_model(x1_0, x2_0)
    print(f"Target instance: (x1={x1_0}, x2={x2_0}), model output = {y_pred:.3f}")

    # 3) Explain with LIME
    a0, a1, a2, linear_model = explainer.explain_instance(x1_0, x2_0,
                                                          num_samples=10,
                                                          noise_scale=0.5)
    print(f"\nLocal Surrogate Model (LIME):\n"
          f"   y_local = {a0:.3f} + {a1:.3f}*x1 + {a2:.3f}*x2")

    # 4) Interpret
    print("\nInterpretation:")
    print(f" - The coefficient for x1 is {a1:.3f}.")
    print(f" - The coefficient for x2 is {a2:.3f}.")
