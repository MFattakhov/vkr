# %%
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from joblib import Parallel, delayed
from scipy.integrate import quad

from analytic_approach import make_u, make_M

# %%
# Parameters
alpha = 1.5
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
# # Plot w0 and w1
# y_vals = np.linspace(-1, 3, 500)
# plt.plot(y_vals, w0(y_vals), label="w0")
# plt.plot(y_vals, w1(y_vals), label="w1")
# plt.legend()
# plt.title("w0 and w1")
# plt.show()
# %%
# Derivatives of w0 and w1
def w0prime(y):
    """Derivative of w0 function - fully vectorized"""
    y = np.asarray(y, dtype=float)
    result = np.zeros_like(y)

    # d/dy(-2*y^3 + 3*y^2) = -6*y^2 + 6*y
    mask1 = (y >= 0) & (y <= 1)
    result[mask1] = -6 * y[mask1] ** 2 + 6 * y[mask1]

    # d/dy(2*y^3 - 9*y^2 + 12*y - 4) = 6*y^2 - 18*y + 12
    mask2 = (y > 1) & (y <= 2)
    result[mask2] = 6 * y[mask2] ** 2 - 18 * y[mask2] + 12

    return result[0] if np.isscalar(y.shape) and y.shape == () else result


def w1prime(y):
    """Derivative of w1 function - fully vectorized"""
    y = np.asarray(y, dtype=float)
    result = np.zeros_like(y)

    # d/dy(y^3 - y^2) = 3*y^2 - 2*y
    mask1 = (y >= 0) & (y <= 1)
    result[mask1] = 3 * y[mask1] ** 2 - 2 * y[mask1]

    # d/dy(y^3 - 5*y^2 + 8*y - 4) = 3*y^2 - 10*y + 8
    mask2 = (y > 1) & (y <= 2)
    result[mask2] = 3 * y[mask2] ** 2 - 10 * y[mask2] + 8

    return result[0] if np.isscalar(y.shape) and y.shape == () else result


# %%
def make_phi0prime(j):
    """Create a shifted derivative of w0 function for specific j value"""

    def phi_prime(x):
        y = np.asarray(x) / h - j
        # Chain rule: d/dx(w0(x/h - j)) = (1/h) * w0'(x/h - j)
        return w0prime(y) / h

    return phi_prime


def make_phi1prime(j):
    """Create a shifted derivative of w1 function for specific j value"""

    def phi_prime(x):
        y = np.asarray(x) / h - j
        # Chain rule: d/dx(w1(x/h - j)) = (1/h) * w1'(x/h - j)
        return w1prime(y) / h

    return phi_prime


# %%
phi0Compiled = [make_phi0(j) for j in range(-1, n)]
phi1Compiled = [make_phi1(j) for j in range(-1, n)]
# %%
phi0D = [make_phi0prime(j) for j in range(-1, n)]
phi1D = [make_phi1prime(j) for j in range(-1, n)]
# %%
size = n + 1


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


# %%
# Integrands
def integrand0(x_val, j, currentAlpha):
    return fc_func(x_val, currentAlpha) * phi0Compiled[j + 1](
        x_val
    )  # j+1 because Python index starts at 0


def integrand1(x_val, j, currentAlpha):
    return fc_func(x_val, currentAlpha) * phi1Compiled[j + 1](x_val)


# %%
# Integration limits
def limits_diag(j):
    return max(0, h * j), min(1, h * (j + 2))


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
f_vec = np.concatenate([f0j, f1j])
# %%
M = make_M(h, alpha)

# %%
def remove_first_row_and_column_from_blocks(M, n):
    # Split M into 4 blocks
    A = M[:n, :n]
    B = M[:n, n:]
    C = M[n:, :n]
    D = M[n:, n:]

    # Remove first row and column from each block
    A_new = A[1:, 1:]
    B_new = B[1:, 1:]
    C_new = C[1:, 1:]
    D_new = D[1:, 1:]

    # Reassemble the new matrix
    M_new = sp.Matrix([[A_new, B_new], [C_new, D_new]])
    return M_new


M_without_1_row_and_col = remove_first_row_and_column_from_blocks(M, M.shape[0] // 2)
M_without_1_row_and_col

# %%
def solve_system(M, f_vec):
    u = sp.Matrix(M.shape[0], 1, lambda i, j: sp.Symbol(f"u_{i + 1}"))
    f_vec_ = sp.zeros(M.shape[0], 1)
    for i in range(M.shape[0]):
        f_vec_[i] = f_vec[i]
    sol = sp.solve(M * u - f_vec_, u)

    return [sol[u[i]] for i in range(M.shape[0])]


A = solve_system(M_without_1_row_and_col, f_vec)


# %%
def u_approx(x):
    # x can be scalar or numpy array
    result = 0
    a_len = len(A)
    for idx in range(len(A) // 2):
        result += A[idx] * phi0Compiled[idx+1](x)
        result += A[idx + a_len // 2] * phi1Compiled[idx+1](x)
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
