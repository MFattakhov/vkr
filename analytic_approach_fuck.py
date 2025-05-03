# %%
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

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

    M_00_diag_first = (
        sp.integrate(
            x ** (2 * alpha) * w0rh_j_p**2
            + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2)) * w0rh_j**2,
            (x, 0, h),
        )
        .subs({j: -1})
        .simplify()
    )
    M_00_subdiag = sp.integrate(
        x ** (2 * alpha) * w0rh_j_p * w0lh_j_p_next
        + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2))
        * w0rh_j
        * w0lh_j_next,
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


# # %%
# f = -((-1 + x) ** 3) + 3 * (-1 + x) * x ** (-1 + alpha) * (-alpha + x * (2 + alpha))
# f = (x ** (3 - alpha) - 2 * (3 - alpha) * x - 1) / (3 - alpha)


# # %%
# def make_f0(h_):
#     n = int(round(1 / h_ - 1))
#     f0 = sp.zeros(n + 1, 1)
#     f0[0] = sp.integrate(f * w0rh_j.subs({h: h_, j: -1}), (x, 0, h_)).simplify()
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
