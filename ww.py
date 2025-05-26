# %%
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from joblib import Parallel, delayed
from scipy.integrate import quad

# from num_approach_fuck import get_M_for_u_numeric, get_M_for_u_prime_numeric
from analytic_approach_fuck_w import get_M_for_u
# from analytic_v2

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


# %%
# Parameters
kappa = 3
sigma = 0.5
sigma2 = sigma**2
theta = 0.04
alpha = 2-2*kappa*theta/sigma2
h = sp.S(1) / 6
n = int(round(1 / h - 1))
# %%
p = sigma2/2 * sp.exp(2 * kappa * x / sigma2)
q = -kappa * x**(1-2*kappa*theta/sigma2)*sp.exp(2 * kappa * x / sigma2)
f = 0
#%%

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
phi0Compiled = [make_phi0(j) for j in range(0, n)]
phi1Compiled = [make_phi1(j) for j in range(0, n)]
# %%
phi0D = [make_phi0prime(j) for j in range(0, n)]
phi1D = [make_phi1prime(j) for j in range(0, n)]
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

    return uexact_expr, uexact_func, fc_expr, fc_func


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

    return uexact_expr, uexact_func, fc_expr, fc_func


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

    return uexact_expr, uexact_func, fc_expr, fc_func


def make_f_and_uexact_4():
    # uexact
    uexact_expr = x ** (3 - alpha) * (1 - x) ** 2
    uexact_func = sp.lambdify((x, a), uexact_expr, "numpy")

    # (-1 + x)^2 x^(3 - α) + 2 x (-3 + 2 x^2 (-5 + α) - 3 x (-4 + α) + α)
    fc_expr = (-1 + x) ** 2 * x ** (3 - alpha) + 2 * x * (
        -3 + 2 * x**2 * (-5 + alpha) - 3 * x * (-4 + alpha) + alpha
    )
    fc_func = sp.lambdify((x, a), fc_expr, "numpy")

    return uexact_expr, uexact_func, fc_expr, fc_func


# %%
# uexact_expr, uexact_func, fc_expr, fc_func = make_f_and_uexact_4()
# uexact_prime = sp.lambdify((x, a), sp.diff(uexact_expr, x), "numpy")
# fc_expr_1 = x**alpha * sp.diff(fc_expr, x)
# fc_func_1 = sp.lambdify((x, a), fc_expr_1, "numpy")

fc_func = sp.lambdify((x, a), 0, "numpy")
fc_func_1 = sp.lambdify((x, a), 0, "numpy")

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
m0 = get_M_for_u(h, alpha)
# m1 = get_M_for_u_prime_numeric(h, alpha)

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
def u_approx_prime(x):
    # x can be scalar or numpy array
    result = 0
    for idx in range(size):
        result += A0[idx] * phi0D[idx](x)
        result += h * A1[idx] * phi1D[idx](x)
    return result


# %%
x_vals = np.linspace(0, 1, 100)
u_approx_vec = np.vectorize(u_approx)
plt.plot(x_vals, u_approx_vec(x_vals), label="Approximate solution $u_{approx}(x)$")
u_approx_prime_vec = np.vectorize(u_approx_prime)
plt.plot(
    x_vals,
    u_approx_prime_vec(x_vals),
    label="Approximate prime of solution $u_{approx}(x)$",
)
plt.plot(
    x_vals,
    uexact_func(x_vals, alpha),
    label="Exact solution $u_{exact}(x)$",
    linestyle="--",
)
plt.plot(
    x_vals,
    uexact_prime(x_vals, alpha),
    label="Exact solution derivative $u_{exact}'(x)$",
    linestyle="--",
)
plt.scatter(
    [float(h * (j + 1)) for j in range(len(A0))],
    A0,
    alpha=0.2,
)
plt.scatter(
    [float(h * (j + 1)) for j in range(len(A0))],
    A1,
    alpha=0.1,
)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Approximate vs Exact Solution")
plt.legend()
plt.show()

# %%
plt.plot(
    x_vals,
    np.abs(u_approx_vec(x_vals) - uexact_func(x_vals, alpha)),
)
plt.show()

# # %%
# for x in np.linspace(0.1, 0.9, 9):
#     print(f"{x:.1f}", f"{np.abs(u_approx(x) - uexact_func(x, alpha)):.2e}")
# # %%
# print(f'({', '.join(f"({v}, {u_approx(v)})" for v in np.linspace(0, 1, 100))})')
# # %%
# print(
#     f'({', '.join(f"({v}, {np.abs(u_approx(v) - uexact_func(v, alpha))})" for v in np.linspace(0, 1, 100))})'
# )

# # %%
# print(
#     f'({', '.join(f"({v}, {u_approx_prime_vec(v)})" for v in np.linspace(0, 1, 100))})'
# )
# # %%
# print(
#     f'({', '.join(f"({v}, {np.abs(u_approx_prime_vec(v) - uexact_prime(v, alpha))})" for v in np.linspace(0, 1, 100))})'
# )

# %%
# Build full approximate vector including boundaries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

n = int(round(1 / h - 1)) + 1
x_nodes = np.linspace(0, 1, n + 1)
u_approx = np.empty(n + 1)
u_approx[1:-1] = np.array(A0)
u_approx[0] = uexact_func(0, alpha)  # or your computed u(0)
u_approx[-1] = uexact_func(1, alpha)  # = 0 in your problem


# --- Method 1: Cubic Hermite Interpolation (slope averaging) ---
def make_hermite(x_nodes, u_nodes, h):
    N = len(u_nodes) - 1
    # compute nodal slopes by centered differences
    d = np.zeros_like(u_nodes)
    d[1:N] = (u_nodes[2:] - u_nodes[:-2]) / (2 * h)
    d[0] = (u_nodes[1] - u_nodes[0]) / h
    d[N] = (u_nodes[N] - u_nodes[N - 1]) / h

    def H(x_eval):
        x_eval = np.atleast_1d(x_eval)
        y = np.zeros_like(x_eval)
        yprime = np.zeros_like(x_eval)
        for j, xv in enumerate(x_eval):
            # locate interval
            if xv <= x_nodes[0]:
                i, t = 0, 0
            elif xv >= x_nodes[-1]:
                i, t = N - 1, 1
            else:
                i = int((xv - x_nodes[0]) // h)
                t = (xv - x_nodes[i]) / h
            u0, u1 = u_nodes[i], u_nodes[i + 1]
            d0, d1 = d[i], d[i + 1]
            # Hermite basis
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            y[j] = h00 * u0 + h10 * h * d0 + h01 * u1 + h11 * h * d1
            # derivatives of basis
            dh00 = 6 * t**2 - 6 * t
            dh10 = 3 * t**2 - 4 * t + 1
            dh01 = -6 * t**2 + 6 * t
            dh11 = 3 * t**2 - 2 * t
            yprime[j] = (dh00 * u0 + dh10 * h * d0 + dh01 * u1 + dh11 * h * d1) / h
        return y, yprime

    return H


H_func = make_hermite(x_nodes, u_approx, h)

# --- Method 2: Natural Cubic Spline (C²) ---
cs = CubicSpline(x_nodes, u_approx, bc_type="natural")

# Evaluate on fine grid
xf = np.linspace(0, 1, 500)
uh_H, up_H = H_func(xf)
uh_CS = cs(xf)
up_CS = cs(xf, 1)
ue = uexact_func(xf,alpha)
upe = uexact_prime(xf,alpha)

def linear_interp(x_nodes, u_nodes, x_eval):
    return np.interp(x_eval, x_nodes, u_nodes)

# Precompute slopes (forward difference)
slopes = np.diff(u_approx) / h

def linear_derivative(x_nodes, slopes, x_eval):
    x_eval = np.atleast_1d(x_eval)
    d = np.zeros_like(x_eval)
    N = len(slopes)
    for j, xv in enumerate(x_eval):
        if xv <= x_nodes[0]:
            i = 0
        elif xv >= x_nodes[-1]:
            i = N - 1
        else:
            i = int((xv - x_nodes[0]) // h)
        d[j] = slopes[i]
    return d

uh_lin = linear_interp(x_nodes, u_approx, xf)
up_lin = linear_derivative(x_nodes, slopes, xf)


# Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(xf, ue, label='Exact')
plt.plot(xf, uh_lin, label='Linear C⁰')
plt.plot(xf, uh_H, '--', label='Hermite C¹')
plt.plot(xf, uh_CS, '-.', label='Cubic Spline C²')
plt.plot(x_nodes, u_approx, 'ko', markersize=3, label='Nodes')
plt.title('Solution Approximation')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(xf, upe, label="Exact $u'$")
plt.plot(xf, up_lin, label="Linear $u'$")
plt.plot(xf, up_H, '--', label="Hermite C¹ $u'$")
plt.plot(xf, up_CS, '-.', label="Cubic Spline C² $u'$")
plt.title('Derivative Approximation')
plt.xlabel('x')
plt.ylabel("u'(x)")
plt.legend()

plt.tight_layout()
plt.show()
# %%
