import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class SHAPHandExplainer(object):
    """
    A basic SHAP explainer for a 2-feature model, e.g. f(x1, x2).
    This class computes Shapley values via enumerating all subsets.
    """

    def __init__(self, model):
        """
        :param model: A function that takes (x1, x2) and returns a probability (0 to 1).
        """
        self.model = model

    def explain_instance(self, x1_0, x2_0, baseline_x1=0.0, baseline_x2=0.0):
        """
        Calculate SHAP values for the instance (x1_0, x2_0) using (baseline_x1, baseline_x2)
        as the baseline. Returns a dictionary with:
            - phi_x1, phi_x2: the Shapley values for x1, x2
            - f_instance: model output at (x1_0, x2_0)
            - f_baseline: model output at (baseline_x1, baseline_x2)
            - check_sum: baseline + phi_x1 + phi_x2 (should match f_instance)
        """
        # 1) Evaluate model at the target instance and baseline
        f_instance = self.model(x1_0, x2_0)
        f_baseline = self.model(baseline_x1, baseline_x2)

        # 2) Enumerate subsets
        # f({}) => both features at baseline
        f_empty = self.model(baseline_x1, baseline_x2)

        # f({x1})
        f_x1 = self.model(x1_0, baseline_x2)

        # f({x2})
        f_x2 = self.model(baseline_x1, x2_0)

        # f({x1, x2}) => target instance
        f_x1x2 = self.model(x1_0, x2_0)

        # 3) Compute Shapley values (phi_x1, phi_x2)
        # phi_x1 = 1/2 * [ (f({x1}) - f({})) + (f({x1,x2}) - f({x2})) ]
        phi_x1 = 0.5 * ((f_x1 - f_empty) + (f_x1x2 - f_x2))

        # phi_x2 = 1/2 * [ (f({x2}) - f({})) + (f({x1,x2}) - f({x1})) ]
        phi_x2 = 0.5 * ((f_x2 - f_empty) + (f_x1x2 - f_x1))

        # 4) Check sum: baseline + sum of Shapley values = final prediction
        check_sum = f_empty + phi_x1 + phi_x2

        return {
            "phi_x1": phi_x1,
            "phi_x2": phi_x2,
            "f_instance": f_instance,
            "f_baseline": f_baseline,
            "check_sum": check_sum
        }


# 1) Define the model we want to explain
def my_model(x1, x2):
    # Example: f(x1, x2) = sigmoid(1.5*x1 - 1.0*x2)
    return sigmoid(1.5 * x1 - x2)

if __name__ == "__main__":
    # 2) Instantiate the explainer
    explainer = SHAPHandExplainer(model=my_model)

    # 3) Define instance and baseline
    x1_0, x2_0 = 2.0, 0.0   # target instance
    baseline_x1, baseline_x2 = 0.0, 0.0  # baseline

    # 4) Run the explainer
    explanation = explainer.explain_instance(x1_0, x2_0,
                                             baseline_x1,
                                             baseline_x2)

    # 5) Print out results
    f_instance = explanation["f_instance"]
    f_baseline = explanation["f_baseline"]
    phi_x1 = explanation["phi_x1"]
    phi_x2 = explanation["phi_x2"]
    check_sum = explanation["check_sum"]

    print(f"Instance: ({x1_0},{x2_0}) => f={f_instance:.3f}")
    print(f"Baseline: ({baseline_x1},{baseline_x2}) => f={f_baseline:.3f}\n")

    print(f"Shapley values:")
    print(f"  phi_x1 = {phi_x1:.3f}")
    print(f"  phi_x2 = {phi_x2:.3f}")

    print(f"\nCheck => baseline + phi_x1 + phi_x2 = {check_sum:.3f}")
    print(f"Model prediction for instance       = {f_instance:.3f}")
