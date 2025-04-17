import numpy as np
x = np.random.randn(3,4)
y = np.random.randn(3,4)
z = np.random.randn(3,4)
a = x * y
b = a + z
c = np.sum(b)
grad_c = 1.0
grad_b = grad_c * np.ones((3,4))
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_x = grad_a * y
grad_y = grad_a *x