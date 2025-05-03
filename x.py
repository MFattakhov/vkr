# %%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from analytic_approach_0_level import make_u

# %%
# Parameters
alpha = 1.8
h = sp.S(1)/4
n = int(round(1 / h - 1))
# %%
# Symbolic variables
x, y, a = sp.symbols("x y a", real=True, positive=True)
# %%
# uexact
uexact_expr = (x**(3-alpha)-1)/(3-alpha)
uexact_func = sp.lambdify(x, uexact_expr, "numpy")


# %%
# %%
# Vectorized w0 and w1 functions
def w(y):
    """Base w0 function - fully vectorized"""
    # Convert input to numpy array
    y = np.asarray(y, dtype=float)

    # Initialize result array with zeros
    result = np.zeros_like(y)

    # Region 0 <= y <= 1
    mask1 = (y >= 0) & (y <= 1)
    result[mask1] = y[mask1]

    # Region 1 < y <= 2
    mask2 = (y > 1) & (y <= 2)
    result[mask2] = 2 - y[mask2]

    # Return scalar if input was scalar
    return result[0] if np.isscalar(y.shape) and y.shape == () else result

# Create function factories for shifted versions
def make_phi(j):
    """Create a shifted w0 function for specific j value"""

    def phi(x):
        # Apply the transformation
        y = np.asarray(x) / h - j
        return w(y)

    return phi
# %%
phiCompiled = [make_phi(j) for j in range(-1, n)]
# %%



A = np.array(make_u(h, alpha))


# %%
size = n + 1


def u_approx(x):
    # x can be scalar or numpy array
    result = 0
    for j in range(-1, n):
        idx = j + 1  # Python index for j in [-1, n-1]
        result += A[idx] * phiCompiled[idx](x)
    return result


# %%
x_vals = np.linspace(0, 1, 100)
u_approx_vec = np.vectorize(u_approx)
plt.plot(x_vals, u_approx_vec(x_vals), label="Approximate solution $u_{approx}(x)$")
plt.plot(
    x_vals, uexact_func(x_vals), label="Exact solution $u_{exact}(x)$", linestyle="--"
)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Approximate vs Exact Solution")
plt.legend()
plt.show()
# %%
