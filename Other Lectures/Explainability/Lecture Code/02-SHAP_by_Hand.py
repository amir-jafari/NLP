import numpy as np
#%%---------------------------------------------------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class SHAPHandExplainer(object):
    def __init__(self, model):
        self.model = model
    def explain_instance(self, x1_0, x2_0, baseline_x1=0.0, baseline_x2=0.0):
        f_instance = self.model(x1_0, x2_0)
        f_baseline = self.model(baseline_x1, baseline_x2)
        f_empty = self.model(baseline_x1, baseline_x2)
        f_x1 = self.model(x1_0, baseline_x2)
        f_x2 = self.model(baseline_x1, x2_0)
        f_x1x2 = self.model(x1_0, x2_0)
        phi_x1 = 0.5 * ((f_x1 - f_empty) + (f_x1x2 - f_x2))
        phi_x2 = 0.5 * ((f_x2 - f_empty) + (f_x1x2 - f_x1))
        check_sum = f_empty + phi_x1 + phi_x2
        return {
            "phi_x1": phi_x1,
            "phi_x2": phi_x2,
            "f_instance": f_instance,
            "f_baseline": f_baseline,
            "check_sum": check_sum
        }

def my_model(x1, x2):
    return sigmoid(1.5 * x1 - x2)
#%%---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    explainer = SHAPHandExplainer(model=my_model)
    x1_0, x2_0 = 2.0, 0.0
    baseline_x1, baseline_x2 = 0.0, 0.0
    explanation = explainer.explain_instance(x1_0, x2_0,
                                             baseline_x1,
                                             baseline_x2)
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
