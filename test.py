# %%
import sympy as sp

# %%
# Define symbolic variables
x, h, y, alpha = sp.symbols("x h y alpha", real=True, positive=True)
j = sp.symbols("j", integer=True)
# %%

# Define basis function w0, w1
wlh = y
wrh = 2 - y

# %%
w0lh_j = wlh.subs({y: (x / h - j)})
w0lh_j_next = wlh.subs({y: (x / h - j - 1)})
w0rh_j = wrh.subs({y: (x / h - j)})
w0rh_j_next = wrh.subs({y: (x / h - j - 1)})
# %%
w0lh_j_p = sp.diff(w0lh_j, x)
w0lh_j_p_next = sp.diff(w0lh_j_next, x)
w0rh_j_p = sp.diff(w0rh_j, x)
w0rh_j_p_next = sp.diff(w0rh_j_next, x)


# %%
def get_M_for_u(h_, alpha_):
    # integrate
    res1 = sp.integrate(
        x**alpha * w0lh_j_p**2 + w0lh_j**2, (x, h * j, h * (j + 1))
    ).simplify()
    res2 = sp.integrate(
        x**alpha * w0rh_j_p**2 + w0rh_j**2, (x, h * (j + 1), h * (j + 2))
    ).simplify()
    M_00_diag_inner = (res1 + res2).simplify()

    M_00_diag_first = (
        sp.integrate(x**alpha * w0rh_j_p**2 + w0rh_j**2, (x, 0, h))
        .subs({j: -1})
        .simplify()
    )
    M_00_subdiag = sp.integrate(
        x**alpha * w0rh_j_p * w0lh_j_p_next + w0rh_j * w0lh_j_next,
        (x, h * (j + 1), h * (j + 2)),
    )

    def make_M_00(h_):
        n = int(round(1 / h_ - 1))
        M_00 = sp.zeros(n + 1, n + 1)
        for j_ in range(-1, n):
            row = j_ + 1
            if row == 0:
                M_00[row, row] = M_00_diag_first.subs({h: h_})
                continue

            M_00[row, row] = M_00_diag_inner.subs({h: h_, j: j_})
            if row != n + 1:
                M_00[row, row - 1] = M_00_subdiag.subs({h: h_, j: j_ - 1})
                M_00[row - 1, row] = M_00_subdiag.subs({h: h_, j: j_ - 1})
        return M_00

    return make_M_00(h_).subs({alpha: alpha_})


# %%
def get_M_for_u_prime(h_, alpha_):
    # integrate
    res1 = sp.integrate(
        x ** (2 * alpha) * w0lh_j_p**2
        + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2)) * w0lh_j**2,
        (x, h * j, h * (j + 1)),
    ).simplify()
    res2 = sp.integrate(
        x ** (2 * alpha) * w0rh_j_p**2
        + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2)) * w0rh_j**2,
        (x, h * (j + 1), h * (j + 2)),
    ).simplify()
    M_00_diag_inner = (res1 + res2).simplify()
    print('here')

    M_00_diag_first = (
        sp.integrate(
            x ** (2 * alpha) * w0rh_j_p**2
            + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2)) * w0rh_j**2,
            (x, sp.S(1) / 2**16, h),
        )
        .subs({j: -1})
        .simplify()
    )
    print('here2')

    M_00_subdiag_first = sp.integrate(
        x ** (2 * alpha) * w0rh_j_p * w0lh_j_p_next
        + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2))
        * w0rh_j
        * w0lh_j_next,
        (x, sp.S(1) / 2**16, h),
    ).subs({j: -1})
    print('here3')
    M_00_subdiag = sp.integrate(
        x ** (2 * alpha) * w0rh_j_p * w0lh_j_p_next
        + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2))
        * w0rh_j
        * w0lh_j_next,
        (x, h * (j + 1), h * (j + 2)),
    )
    print('here4')

    def make_M_00(h_):
        n = int(round(1 / h_ - 1))
        M_00 = sp.zeros(n + 1, n + 1)
        for j_ in range(-1, n):
            print(f'here {j_=}')

            row = j_ + 1
            if row == 0:
                M_00[row, row] = M_00_diag_first.subs({h: h_})
                continue

            M_00[row, row] = M_00_diag_inner.subs({h: h_, j: j_})
            if row != n + 1:
                if row != 1:
                    M_00[row, row - 1] = M_00_subdiag.subs({h: h_, j: j_ - 1})
                    M_00[row - 1, row] = M_00_subdiag.subs({h: h_, j: j_ - 1})
                else:
                    M_00[row, row - 1] = M_00_subdiag_first.subs({h: h_})
                    M_00[row - 1, row] = M_00_subdiag_first.subs({h: h_})
        return M_00

    return make_M_00(h_).subs({alpha: alpha_})


# # %%
# f = -((-1 + x) ** 3) + 3 * (-1 + x) * x ** (-1 + alpha) * (-alpha + x * (2 + alpha))
# f = (x ** (3 - alpha) - 2 * (3 - alpha) * x - 1) / (3 - alpha)


# # %%
# def make_f0(h_):
#     n = int(round(1 / h_ - 1))
#     f0 = sp.zeros(n + 1, 1)
#     f0[0] = sp.integrate(f * w0rh_j.subs({h: h_, j: -1}), (x, sp.S(1)/2**16, h_)).simplify()
#     for j_ in range(0, n):
#         f0[j_ + 1] = (
#             sp.integrate(f * w0lh_j.subs({h: h_, j: j_}), (x, h_ * j_, h_ * (j_ + 1)))
#             + sp.integrate(
#                 f * w0rh_j.subs({h: h_, j: j_}), (x, h_ * (j_ + 1), h_ * (j_ + 2))
#             )
#         ).simplify()
#     return f0


# # %%
# def make_f(h_):
#     return make_f0(h_)


# # %%
# def make_u(h_, alpha_):
#     f_vec = make_f(h_).subs({alpha: alpha_})
#     print(f"{f_vec=}")
#     M = make_M(h_).subs({alpha: alpha_})

#     u = sp.Matrix(M.shape[0], 1, lambda i, j: sp.Symbol(f"u_{i + 1}"))
#     sol = sp.solve(M * u - f_vec, u)

#     return [sol[u[i]] for i in range(M.shape[0])]


# %%
# make_f(0.25).subs({alpha: 1})
# %%

# # Convert to numerical functions
# h_val = 0.25
# j_val = 1
# alpha_val = 1  # Not used in these functions

# w0lh_j_num = sp.lambdify(x, w0lh_j.subs({h: h_val, j: j_val}), 'numpy')
# w0rh_j_num = sp.lambdify(x, w0rh_j.subs({h: h_val, j: j_val}), 'numpy')

# # Create x values for plotting
# x_vals_lh = np.linspace(j_val * h_val, (j_val + 1) * h_val, 100)
# x_vals_rh = np.linspace((j_val+1) * h_val, (j_val + 2) * h_val, 100)

# # Evaluate functions
# y_lh = w0lh_j_num(x_vals_lh)
# y_rh = w0rh_j_num(x_vals_rh)

# # Plot
# plt.figure(figsize=(10, 6))
# plt.plot(x_vals_lh, y_lh, label='w0lh_j')
# plt.plot(x_vals_rh, y_rh, label='w0rh_j')
# plt.axvline(x=j_val*h_val, color='gray', linestyle='--')
# plt.axvline(x=(j_val+1)*h_val, color='gray', linestyle='--')
# plt.title(f'Basis functions with h={h_val}, j={j_val}')
# plt.xlabel('x')
# plt.ylabel('Function value')
# plt.legend()
# plt.grid(True)
# plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from joblib import Parallel, delayed
from scipy.integrate import quad



# %%
# Parameters
alpha = 1
h = sp.S(1) / 6
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
phi0Compiled = [make_phi0(j) for j in range(-1, n)]
phi1Compiled = [make_phi1(j) for j in range(-1, n)]
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
uexact_expr, uexact_func, fc_expr, fc_func = make_f_and_uexact_1()
uexact_prime = sp.lambdify((x, a), sp.diff(uexact_expr, x), "numpy")
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
    return max(0, h * j), h * (j + 2)


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
m0 = get_M_for_u(h, alpha)
#%%
m1 = get_M_for_u_prime(h, alpha)


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
#%%
m1

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
plt.plot(
    x_vals,
    uexact_prime(x_vals, alpha),
    label="Exact solution derivative $u_{exact}'(x)$",
    linestyle="--",
)
plt.scatter(
    [float(h * j) for j in range(len(A0))],
    A0,
    alpha=0.2,
)
plt.scatter(
    [float(h * j) for j in range(len(A0))],
    A1,
    alpha=0.1,
)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Approximate vs Exact Solution")
plt.legend()
plt.show()

# %%
