# %%
import numpy as np
import sympy as sp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# %%
# Parameters
alpha = 1
h = 0.2
n = int(round(1 / h - 1))
# %%
# Symbolic variables
x, y, a = sp.symbols("x y a", real=True, positive=True)
# %%
# fc: the main function (symbolic)
fc_expr = (
    -(
        sp.exp(-x)
        * (1 - x)
        * (-3 * x + x**2 + sp.exp(x) * x**a * a - sp.exp(x) * x ** (1 + a) * (2 + a))
    )
    / x
)
fc_func = sp.lambdify((x, a), fc_expr, "numpy")
# %%
# For plotting f(x, alpha)
x_vals = np.linspace(0.0001, 1, 500)
f_vals = fc_func(x_vals, alpha)
plt.plot(x_vals, f_vals)
plt.title("f(x, alpha)")
plt.xlabel("x")
plt.ylabel("f(x, alpha)")
plt.show()
# %%
# uexact
uexact_expr = sp.exp(-x) * (1 - x) ** 2
uexact_func = sp.lambdify(x, uexact_expr, "numpy")
# %%
# w0 and w1 as piecewise functions
w0_expr = sp.Piecewise(
    (-2 * y**3 + 3 * y**2, (y >= 0) & (y <= 1)),
    (2 * y**3 - 9 * y**2 + 12 * y - 4, (y > 1) & (y <= 2)),
    (0, True),
)
w1_expr = sp.Piecewise(
    (y**3 - y**2, (y >= 0) & (y <= 1)),
    (y**3 - 5 * y**2 + 8 * y - 4, (y > 1) & (y <= 2)),
    (0, True),
)
w0_func = sp.lambdify(y, w0_expr, "numpy")
w1_func = sp.lambdify(y, w1_expr, "numpy")
# %%
# Plot w0 and w1
y_vals = np.linspace(0, 2, 500)
plt.plot(y_vals, w0_func(y_vals), label="w0")
plt.plot(y_vals, w1_func(y_vals), label="w1")
plt.legend()
plt.title("w0 and w1")
plt.show()
# %%
# phi0Compiled and phi1Compiled
phi0Compiled = [
    sp.lambdify(x, w0_expr.subs(y, x / h - j), "numpy") for j in range(-1, n)
]
phi1Compiled = [
    sp.lambdify(x, w1_expr.subs(y, x / h - j), "numpy") for j in range(-1, n)
]
# %%
# Derivatives
w0prime_expr = sp.diff(w0_expr, y)
w1prime_expr = sp.diff(w1_expr, y)
# %%
# phi0D and phi1D
phi0D = [
    sp.lambdify(x, (1 / h) * w0prime_expr.subs(y, x / h - j), "numpy")
    for j in range(-1, n)
]
phi1D = [
    sp.lambdify(x, (1 / h) * w1prime_expr.subs(y, x / h - j), "numpy")
    for j in range(-1, n)
]


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
f0j = Parallel(n_jobs=-1)(delayed(integrate_f0j)(j) for j in range(-1, n))
f1j = Parallel(n_jobs=-1)(delayed(integrate_f1j)(j) for j in range(-1, n))


# %%
# integrandT00: x, j, alpha
def integrandT00(x_val, j, alpha):
    # j+1 for Python 0-based indexing
    phi0Val = phi0Compiled[j + 1](x_val)
    phi0DVal = phi0D[j + 1](x_val)
    return np.exp(-x_val) * x_val**alpha * phi0DVal**2 + phi0Val**2


# %%
# Integration with exclusion if needed
def integrate_t00(j):
    a_, b_ = limits_diag(j)
    # Exclude x=0 only if it's the lower bound and alpha <= 1
    if a_ == 0 and alpha <= 1:
        # Integrate from a small positive number instead of 0
        a_ = 1e-12
    result, _ = quad(
        integrandT00, a_, b_, args=(j, alpha), epsabs=1e-8, epsrel=1e-8, limit=100
    )
    return result


# %%
t00 = Parallel(n_jobs=-1)(delayed(integrate_t00)(j) for j in range(-1, n))


# %%
# [phi0_j, phi0_{j+1}] inner product integrand
def integrandT00_subdiag(x_val, j, alpha):
    phi0j = phi0Compiled[j + 1](x_val)
    phi0jp1 = phi0Compiled[j + 2](x_val)
    phi0Dj = phi0D[j + 1](x_val)
    phi0Djp1 = phi0D[j + 2](x_val)
    return np.exp(-x_val) * x_val**alpha * phi0Dj * phi0Djp1 + phi0j * phi0jp1


# %%
# Integration limits: overlap of supports of phi0_j and phi0_{j+1}
def limits_subdiag(j):
    # phi0_j is nonzero on [h*j, h*(j+2)]
    # phi0_{j+1} is nonzero on [h*(j+1), h*(j+3)]
    a = max(0, h * (j + 1))
    b = min(1, h * (j + 2))
    return a, b


# %%
def integrate_t00_subdiag(j):
    a_, b_ = limits_subdiag(j)
    # Exclude x=0 only if it's the lower bound and alpha <= 1
    if a_ == 0 and alpha <= 1:
        a_ = 1e-12
    if b_ <= a_:
        return 0.0
    result, _ = quad(
        integrandT00_subdiag,
        a_,
        b_,
        args=(j, alpha),
        epsabs=1e-8,
        epsrel=1e-8,
        limit=100,
    )
    return result


# %%
# Compute subdiagonal for j = -1 to n-2 (since j+1 must be <= n-1)
t00_subdiag = Parallel(n_jobs=-1)(
    delayed(integrate_t00_subdiag)(j) for j in range(-1, n - 1)
)
# %%
size = n + 1
M_00 = np.zeros((size, size))

# Fill diagonal
for i in range(size):
    M_00[i, i] = t00[i]

# Fill subdiagonal and superdiagonal
for i in range(size - 1):
    M_00[i, i + 1] = t00_subdiag[i]
    M_00[i + 1, i] = t00_subdiag[i]


# %%
def integrand_01_diag(x_val, j, alpha):
    val0 = phi0Compiled[j + 1](x_val)
    val0D = phi0D[j + 1](x_val)
    val1 = phi1Compiled[j + 1](x_val)
    val1D = phi1D[j + 1](x_val)
    return np.exp(-x_val) * x_val**alpha * val0D * val1D + val0 * val1


def integrand_01_subdiag(x_val, j, alpha):
    val0 = phi0Compiled[j + 1](x_val)
    val0D = phi0D[j + 1](x_val)
    val1 = phi1Compiled[j + 2](x_val)
    val1D = phi1D[j + 2](x_val)
    return np.exp(-x_val) * x_val**alpha * val0D * val1D + val0 * val1


def integrate_01_diag(j):
    a_, b_ = limits_diag(j)
    if a_ == 0 and alpha <= 1:
        a_ = 1e-12
    if b_ <= a_:
        return 0.0
    result, _ = quad(
        integrand_01_diag, a_, b_, args=(j, alpha), epsabs=1e-8, epsrel=1e-8, limit=100
    )
    return result


def integrate_01_subdiag(j):
    a_, b_ = limits_subdiag(j)
    if a_ == 0 and alpha <= 1:
        a_ = 1e-12
    if b_ <= a_:
        return 0.0
    result, _ = quad(
        integrand_01_subdiag,
        a_,
        b_,
        args=(j, alpha),
        epsabs=1e-8,
        epsrel=1e-8,
        limit=100,
    )
    return result


# Diagonal
M_01_diag = Parallel(n_jobs=-1)(delayed(integrate_01_diag)(j) for j in range(-1, n))

# Subdiagonal
M_01_subdiag = Parallel(n_jobs=-1)(
    delayed(integrate_01_subdiag)(j) for j in range(-1, n - 1)
)

# Assemble banded matrix
M_01 = np.zeros((size, size))
for i, val in enumerate(M_01_diag):
    M_01[i, i] = val
for i, val in enumerate(M_01_subdiag):
    M_01[i, i + 1] = val
    M_01[i + 1, i] = val
# %%
def integrand_10_diag(x_val, j, alpha):
    val1 = phi1Compiled[j+1](x_val)
    val1D = phi1D[j+1](x_val)
    val0 = phi0Compiled[j+1](x_val)
    val0D = phi0D[j+1](x_val)
    return np.exp(-x_val) * x_val**alpha * val1D * val0D + val1 * val0

def integrand_10_subdiag(x_val, j, alpha):
    val1 = phi1Compiled[j+1](x_val)
    val1D = phi1D[j+1](x_val)
    val0 = phi0Compiled[j+2](x_val)
    val0D = phi0D[j+2](x_val)
    return np.exp(-x_val) * x_val**alpha * val1D * val0D + val1 * val0

def integrate_10_diag(j):
    a_, b_ = limits_diag(j)
    if a_ == 0 and alpha <= 1:
        a_ = 1e-12
    if b_ <= a_:
        return 0.0
    result, _ = quad(integrand_10_diag, a_, b_, args=(j, alpha), epsabs=1e-8, epsrel=1e-8, limit=100)
    return result

def integrate_10_subdiag(j):
    a_, b_ = limits_subdiag(j)
    if a_ == 0 and alpha <= 1:
        a_ = 1e-12
    if b_ <= a_:
        return 0.0
    result, _ = quad(integrand_10_subdiag, a_, b_, args=(j, alpha), epsabs=1e-8, epsrel=1e-8, limit=100)
    return result

M_10_diag = Parallel(n_jobs=-1)(
    delayed(integrate_10_diag)(j) for j in range(-1, n)
)
M_10_subdiag = Parallel(n_jobs=-1)(
    delayed(integrate_10_subdiag)(j) for j in range(-1, n-1)
)

M_10 = np.zeros((size, size))
for i, val in enumerate(M_10_diag):
    M_10[i, i] = val
for i, val in enumerate(M_10_subdiag):
    M_10[i, i+1] = val
    M_10[i+1, i] = val
# %%
def integrand_11_diag(x_val, j, alpha):
    val1 = phi1Compiled[j+1](x_val)
    val1D = phi1D[j+1](x_val)
    return np.exp(-x_val) * x_val**alpha * val1D**2 + val1**2

def integrand_11_subdiag(x_val, j, alpha):
    val1 = phi1Compiled[j+1](x_val)
    val1D = phi1D[j+1](x_val)
    val1p = phi1Compiled[j+2](x_val)
    val1Dp = phi1D[j+2](x_val)
    return np.exp(-x_val) * x_val**alpha * val1D * val1Dp + val1 * val1p

def integrate_11_diag(j):
    a_, b_ = limits_diag(j)
    if a_ == 0 and alpha <= 1:
        a_ = 1e-12
    if b_ <= a_:
        return 0.0
    result, _ = quad(integrand_11_diag, a_, b_, args=(j, alpha), epsabs=1e-8, epsrel=1e-8, limit=100)
    return result

def integrate_11_subdiag(j):
    a_, b_ = limits_subdiag(j)
    if a_ == 0 and alpha <= 1:
        a_ = 1e-12
    if b_ <= a_:
        return 0.0
    result, _ = quad(integrand_11_subdiag, a_, b_, args=(j, alpha), epsabs=1e-8, epsrel=1e-8, limit=100)
    return result

M_11_diag = Parallel(n_jobs=-1)(
    delayed(integrate_11_diag)(j) for j in range(-1, n)
)
M_11_subdiag = Parallel(n_jobs=-1)(
    delayed(integrate_11_subdiag)(j) for j in range(-1, n-1)
)

M_11 = np.zeros((size, size))
for i, val in enumerate(M_11_diag):
    M_11[i, i] = val
for i, val in enumerate(M_11_subdiag):
    M_11[i, i+1] = val
    M_11[i+1, i] = val
# %%
# Scalar product integrand for (f, w0_j)
def integrand_b0(x_val, j, alpha):
    return fc_func(x_val, alpha) * phi0Compiled[j+1](x_val)

# Scalar product integrand for (f, w1_j)
def integrand_b1(x_val, j, alpha):
    return fc_func(x_val, alpha) * phi1Compiled[j+1](x_val)

def limits_b(j):
    # w*_j is nonzero on [h*j, h*(j+2)]
    a = max(0, h*j)
    b = min(1, h*(j+2))
    return a, b

def integrate_b(integrand, j):
    a_, b_ = limits_b(j)
    if a_ == 0 and alpha <= 1:
        a_ = 1e-12
    if b_ <= a_:
        return 0.0
    result, _ = quad(integrand, a_, b_, args=(j, alpha), epsabs=1e-8, epsrel=1e-8, limit=100)
    return result

# Compute both vectors in parallel
b0_vec = Parallel(n_jobs=-1)(
    delayed(integrate_b)(integrand_b0, j) for j in range(-1, n)
)
b1_vec = Parallel(n_jobs=-1)(
    delayed(integrate_b)(integrand_b1, j) for j in range(-1, n)
)

# Concatenate
b_full = np.concatenate([b0_vec, b1_vec])
# %%
# Assemble the full block matrix
M = np.zeros((2*size, 2*size))
M[:size, :size] = M_00
M[:size, size:] = M_01
M[size:, :size] = M_10
M[size:, size:] = M_11
# %%
# Solve the linear system
A = np.linalg.solve(M, b_full)
# %%

def u_approx(x):
    # x can be scalar or numpy array
    result = 0
    for j in range(-1, n):
        idx = j + 1  # Python index for j in [-1, n-1]
        result += A[idx] * phi0Compiled[idx](x)
        result += A[idx + size] * phi1Compiled[idx](x)
    return result
# %%
# Vectorized version for plotting
u_approx_vec = np.vectorize(u_approx)

x_vals = np.linspace(0, 1, 400)
u_vals = u_approx_vec(x_vals)

plt.plot(x_vals, u_vals, label='u_approx(x)')
plt.xlabel('x')
plt.ylabel('u_approx(x)')
plt.title('Approximate Solution')
plt.legend()
plt.show()
# %%
plt.plot(x_vals, u_approx_vec(x_vals), label='Approximate solution $u_{approx}(x)$')
plt.plot(x_vals, uexact_func(x_vals), label='Exact solution $u_{exact}(x)$', linestyle='--')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Approximate vs Exact Solution')
plt.legend()
plt.show()
# %%
