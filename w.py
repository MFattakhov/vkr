# %%
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from joblib import Parallel, delayed
from scipy.integrate import quad

from num_approach_fuck import get_M_for_u_numeric, get_M_for_u_prime_numeric

# %%
# Parameters
alpha = 1
h = sp.S(1) / 8
n = int(round(1 / h - 1))
# %%
# Symbolic variables
x, y, a = sp.symbols("x y a", real=True, positive=True)


# %%
# Vectorized w0 and w1 functions
def w0(y):
    """Base w0 function - fully vectorized"""
    # Convert input to numpy array
    y = np.asarray(y, dtype=float)

    # Initialize result array with zeros
    result = np.zeros_like(y)

    # Region 0 <= y <= 1
    mask1 = (y >= 0) & (y <= 1)
    result[mask1] = -2 * y[mask1] ** 3 + 3 * y[mask1] ** 2

    # Region 1 < y <= 2
    mask2 = (y > 1) & (y <= 2)
    result[mask2] = 2 * y[mask2] ** 3 - 9 * y[mask2] ** 2 + 12 * y[mask2] - 4

    # Return scalar if input was scalar
    return result[0] if np.isscalar(y.shape) and y.shape == () else result


def w1(y):
    """Base w1 function - fully vectorized"""
    # Convert input to numpy array
    y = np.asarray(y, dtype=float)

    # Initialize result array with zeros
    result = np.zeros_like(y)

    # Region 0 <= y <= 1
    mask1 = (y >= 0) & (y <= 1)
    result[mask1] = y[mask1] ** 3 - y[mask1] ** 2

    # Region 1 < y <= 2
    mask2 = (y > 1) & (y <= 2)
    result[mask2] = y[mask2] ** 3 - 5 * y[mask2] ** 2 + 8 * y[mask2] - 4

    # Return scalar if input was scalar
    return result[0] if np.isscalar(y.shape) and y.shape == () else result


# Create function factories for shifted versions
def make_phi0(j):
    """Create a shifted w0 function for specific j value"""

    def phi(x):
        # Apply the transformation
        y = np.asarray(x) / h - j
        return w0(y)

    return phi


def make_phi1(j):
    """Create a shifted w1 function for specific j value"""

    def phi(x):
        # Apply the transformation
        y = np.asarray(x) / h - j
        return w1(y)

    return phi


# %%
phi0Compiled = [make_phi0(j) for j in range(0, n)]
phi1Compiled = [make_phi1(j) for j in range(0, n)]
# %%
size = n


# %%
def make_f_and_uexact_1():
    # uexact
    uexact_expr = (1 - x) ** 3
    uexact_func = sp.lambdify((x, a), uexact_expr, "numpy")

    # 3 (x - 1) x^(α - 1) ((α + 2) x - α) - (x - 1)^3
    fc_expr = 3 * (x - 1) * x ** (alpha - 1) * ((alpha + 2) * x - alpha) - (x - 1) ** 3
    fc_func = sp.lambdify((x, a), fc_expr, "numpy")

    return uexact_func, fc_func


def make_f_and_uexact_2():
    # uexact
    uexact_expr = ((1 - x) ** (3 - alpha)) / (3 - alpha)
    uexact_func = sp.lambdify((x, a), uexact_expr, "numpy")

    # ((1 - x)^(-α) ((-1 + x)^3 x + (-1 + x) x^α (2 x - α) (-3 + α)))/(x (-3 + α))
    fc_expr = (
        (1 - x) ** (-alpha)
        * ((-1 + x) ** 3 * x + (-1 + x) * x**alpha * (2 * x - alpha) * (-3 + alpha))
    ) / (x * (-3 + alpha))
    fc_func = sp.lambdify((x, a), fc_expr, "numpy")

    return uexact_func, fc_func


def make_f_and_uexact_3():
    # uexact
    uexact_expr = -((1 - x) ** (3 - alpha)) / (3 - alpha)
    uexact_func = sp.lambdify((x, a), uexact_expr, "numpy")

    # ((1 - x)^(-α) ((-1 + x)^3 x + (-1 + x) x^α (2 x - α) (-3 + α)))/(x (-3 + α))
    fc_expr = -(
        (1 - x) ** (-alpha)
        * ((-1 + x) ** 3 * x + (-1 + x) * x**alpha * (2 * x - alpha) * (-3 + alpha))
    ) / (x * (-3 + alpha))
    fc_func = sp.lambdify((x, a), fc_expr, "numpy")

    return uexact_func, fc_func


def make_f_and_uexact_4():
    # uexact
    uexact_expr = x ** (3 - alpha) * (1 - x) ** 2
    uexact_func = sp.lambdify((x, a), uexact_expr, "numpy")

    # (-1 + x)^2 x^(3 - α) + 2 x (-3 + 2 x^2 (-5 + α) - 3 x (-4 + α) + α)
    fc_expr = (-1 + x) ** 2 * x ** (3 - alpha) + 2 * x * (
        -3 + 2 * x**2 * (-5 + alpha) - 3 * x * (-4 + alpha) + alpha
    )
    fc_func = sp.lambdify((x, a), fc_expr, "numpy")

    return uexact_func, fc_func


# %%
uexact_func, fc_func = make_f_and_uexact_4()
fc_expr = (-1 + x) ** 2 * x ** (3 - alpha) + 2 * x * (
    -3 + 2 * x**2 * (-5 + alpha) - 3 * x * (-4 + alpha) + alpha
)
fc_expr_1 = x**alpha * sp.diff(fc_expr, x)
fc_func_1 = sp.lambdify((x, a), fc_expr_1, "numpy")


# %%
# Integrands
def integrand0(x_val, j, currentAlpha):
    return fc_func(x_val, currentAlpha) * phi0Compiled[j](
        x_val
    )  # j+1 because Python index starts at 0


def integrand1(x_val, j, currentAlpha):
    return fc_func_1(x_val, currentAlpha) * phi0Compiled[j](x_val)


# %%
# Integration limits
def limits_diag(j):
    return h * j, h * (j + 2)


# %%
# Parallel integration
def integrate_f0j(j):
    a_, b_ = limits_diag(j)
    result, _ = quad(
        integrand0, a_, b_, args=(j, alpha), epsabs=1e-8, epsrel=1e-8, limit=100
    )
    return result


def integrate_f1j(j):
    a_, b_ = limits_diag(j)
    result, _ = quad(
        integrand1, a_, b_, args=(j, alpha), epsabs=1e-8, epsrel=1e-8, limit=100
    )
    return result


# %%
f0j = Parallel(n_jobs=-1)(delayed(integrate_f0j)(j) for j in range(0, n))
f1j = Parallel(n_jobs=-1)(delayed(integrate_f1j)(j) for j in range(0, n))
# %%
m0 = get_M_for_u_numeric(h, alpha)
m1 = get_M_for_u_prime_numeric(h, alpha)

# %%
m0 = m0[1:, 1:]
m1 = m1[1:, 1:]


# %%
def solve_system(M, f_vec):
    u = sp.Matrix(M.shape[0], 1, lambda i, j: sp.Symbol(f"u_{i + 1}"))
    f_vec_ = sp.zeros(M.shape[0], 1)
    for i in range(M.shape[0]):
        f_vec_[i] = f_vec[i]
    sol = sp.solve(M * u - f_vec_, u)

    return [sol[u[i]] for i in range(M.shape[0])]


A0 = solve_system(m0, f0j)
A1 = solve_system(m1, f1j)


# %%
def u_approx(x):
    # x can be scalar or numpy array
    result = 0
    for idx in range(size):
        result += A0[idx] * phi0Compiled[idx](x)
        result += h * A1[idx] * phi1Compiled[idx](x)
    return result


# %%
x_vals = np.linspace(0, 1, 100)
u_approx_vec = np.vectorize(u_approx)
plt.plot(x_vals, u_approx_vec(x_vals), label="Approximate solution $u_{approx}(x)$")
plt.plot(
    x_vals,
    uexact_func(x_vals, alpha),
    label="Exact solution $u_{exact}(x)$",
    linestyle="--",
)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Approximate vs Exact Solution")
plt.legend()
plt.show()

# %%
