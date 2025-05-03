# %%
import sympy as sp

# %%
# Define symbolic variables
x, h, y, alpha = sp.symbols("x h y alpha", real=True, positive=True)
j = sp.symbols("j", integer=True)
# %%

# Define basis function w0, w1
w0lh = -2 * y**3 + 3 * y**2
w0rh = 2 * y**3 - 9 * y**2 + 12 * y - 4

w1lh = y**3 - y**2
w1rh = y**3 - 5 * y**2 + 8 * y - 4

# %%
w0lh_j = w0lh.subs({y: (x / h - j)})
w0lh_j_next = w0lh.subs({y: (x / h - j - 1)})
w0rh_j = w0rh.subs({y: (x / h - j)})
w0rh_j_next = w0rh.subs({y: (x / h - j - 1)})

w1lh_j = w1lh.subs({y: (x / h - j)})
w1lh_j_next = w1lh.subs({y: (x / h - j - 1)})
w1rh_j = w1rh.subs({y: (x / h - j)})
w1rh_j_next = w1rh.subs({y: (x / h - j - 1)})
# %%
w0lh_j_p = sp.diff(w0lh_j, x)
w0lh_j_p_next = sp.diff(w0lh_j_next, x)
w0rh_j_p = sp.diff(w0rh_j, x)
w0rh_j_p_next = sp.diff(w0rh_j_next, x)

w1lh_j_p = sp.diff(w1lh_j, x)
w1lh_j_p_next = sp.diff(w1lh_j_next, x)
w1rh_j_p = sp.diff(w1rh_j, x)
w1rh_j_p_next = sp.diff(w1rh_j_next, x)
# %%
# integrate
res1 = sp.integrate(
    x**alpha * w0lh_j_p**2 + w0lh_j**2, (x, h * j, h * (j + 1))
).simplify()
res2 = sp.integrate(
    x**alpha * w0rh_j_p**2 + w0rh_j**2, (x, h * (j + 1), h * (j + 2))
).simplify()
# %%
M_00_diag_inner = (res1 + res2).simplify()

# %%
M_00_diag_first = (
    sp.integrate(x**alpha * w0rh_j_p**2 + w0rh_j**2, (x, 0, h)).subs({j: -1}).simplify()
)
# %%
M_00_subdiag = sp.integrate(
    x**alpha * w0rh_j_p * w0lh_j_p_next + w0rh_j * w0lh_j_next,
    (x, h * (j + 1), h * (j + 2)),
)


# %%
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


# %%
make_M_00(0.25).subs({alpha: 1})
# %%
# integrate
res1 = sp.integrate(
    x**alpha * w1lh_j_p**2 + w1lh_j**2, (x, h * j, h * (j + 1))
).simplify()
res2 = sp.integrate(
    x**alpha * w1rh_j_p**2 + w1rh_j**2, (x, h * (j + 1), h * (j + 2))
).simplify()
# %%
M_11_diag_inner = (res1 + res2).simplify()

# %%
M_11_diag_first = (
    sp.integrate(x**alpha * w1rh_j_p**2 + w1rh_j**2, (x, 0, h)).subs({j: -1}).simplify()
)
# %%
M_11_subdiag = sp.integrate(
    x**alpha * w1rh_j_p * w1lh_j_p_next + w1rh_j * w1lh_j_next,
    (x, h * (j + 1), h * (j + 2)),
)


# %%
def make_M_11(h_):
    n = int(round(1 / h_ - 1))
    M_11 = sp.zeros(n + 1, n + 1)
    for j_ in range(-1, n):
        row = j_ + 1
        if row == 0:
            M_11[row, row] = M_11_diag_first.subs({h: h_})
            continue

        M_11[row, row] = M_11_diag_inner.subs({h: h_, j: j_})
        if row != n + 1:
            M_11[row, row - 1] = M_11_subdiag.subs({h: h_, j: j_ - 1})
            M_11[row - 1, row] = M_11_subdiag.subs({h: h_, j: j_ - 1})
    return M_11


# %%
# integrate
res1 = sp.integrate(
    x**alpha * w0lh_j_p * w1lh_j_p + w0lh_j * w1lh_j, (x, h * j, h * (j + 1))
).simplify()
res2 = sp.integrate(
    x**alpha * w0rh_j_p * w1rh_j_p + w0rh_j * w1rh_j, (x, h * (j + 1), h * (j + 2))
).simplify()
# %%
M_01_diag_inner = (res1 + res2).simplify()

# %%
M_01_diag_first = (
    sp.integrate(x**alpha * w0rh_j_p * w1rh_j_p + w0rh_j * w1rh_j, (x, 0, h))
    .subs({j: -1})
    .simplify()
)
# %%
M_01_subdiag = sp.integrate(
    x**alpha * w0rh_j_p * w1lh_j_p_next + w0rh_j * w1lh_j_next,
    (x, h * (j + 1), h * (j + 2)),
)


# %%
def make_M_01(h_):
    n = int(round(1 / h_ - 1))
    M_01 = sp.zeros(n + 1, n + 1)
    for j_ in range(-1, n):
        row = j_ + 1
        if row == 0:
            M_01[row, row] = M_01_diag_first.subs({h: h_})
            continue

        M_01[row, row] = M_01_diag_inner.subs({h: h_, j: j_})
        if row != n + 1:
            M_01[row, row - 1] = M_01_subdiag.subs({h: h_, j: j_ - 1})
            M_01[row - 1, row] = M_01_subdiag.subs({h: h_, j: j_ - 1})
    return M_01


# %%
# integrate
res1 = sp.integrate(
    x**alpha * w1lh_j_p * w0lh_j_p + w1lh_j * w0lh_j, (x, h * j, h * (j + 1))
).simplify()
res2 = sp.integrate(
    x**alpha * w1rh_j_p * w0rh_j_p + w1rh_j * w0rh_j, (x, h * (j + 1), h * (j + 2))
).simplify()
# %%
M_10_diag_inner = (res1 + res2).simplify()

# %%
M_10_diag_first = (
    sp.integrate(x**alpha * w1rh_j_p * w0rh_j_p + w1rh_j * w0rh_j, (x, 0, h))
    .subs({j: -1})
    .simplify()
)
# %%
M_10_subdiag = sp.integrate(
    x**alpha * w1rh_j_p * w0lh_j_p_next + w1rh_j * w0lh_j_next,
    (x, h * (j + 1), h * (j + 2)),
)


# %%
def make_M_10(h_):
    n = int(round(1 / h_ - 1))
    M_10 = sp.zeros(n + 1, n + 1)
    for j_ in range(-1, n):
        row = j_ + 1
        if row == 0:
            M_10[row, row] = M_10_diag_first.subs({h: h_})
            continue

        M_10[row, row] = M_10_diag_inner.subs({h: h_, j: j_})
        if row != n + 1:
            M_10[row, row - 1] = M_10_subdiag.subs({h: h_, j: j_ - 1})
            M_10[row - 1, row] = M_10_subdiag.subs({h: h_, j: j_ - 1})
    return M_10


# %%
def make_M(h_, alpha_):
    n = int(round(1 / h_ - 1))
    M_00 = make_M_00(h_)
    M_01 = make_M_01(h_)
    M_10 = make_M_10(h_)
    M_11 = make_M_11(h_)
    M = sp.zeros(2 * n + 2, 2 * n + 2)
    for i_ in range(n + 1):
        for j_ in range(n + 1):
            M[i_, j_] = M_00[i_, j_]
            M[i_ + n + 1, j_] = M_10[i_, j_]
            M[i_, j_ + n + 1] = M_01[i_, j_]
            M[i_ + n + 1, j_ + n + 1] = M_11[i_, j_]
    return M.subs({alpha: alpha_})


# f vec ------------------------------------------------------
# %%
f = -((-1 + x) ** 3) + 3 * (-1 + x) * x ** (-1 + alpha) * (-alpha + x * (2 + alpha))


# %%
# Integration limits
def limits_diag(j):
    return max(0, h * j), min(1, h * (j + 2))


# %%
def make_f0(h_):
    n = int(round(1 / h_ - 1))
    f0 = sp.zeros(n + 1, 1)
    f0[0] = sp.integrate(f * w0rh_j.subs({h: h_, j: -1}), (x, 0, h_)).simplify()
    for j_ in range(0, n):
        f0[j_ + 1] = (
            sp.integrate(f * w0lh_j.subs({h: h_, j: j_}), (x, h_ * j_, h_ * (j_ + 1)))
            + sp.integrate(
                f * w0rh_j.subs({h: h_, j: j_}), (x, h_ * (j_ + 1), h_ * (j_ + 2))
            )
        ).simplify()
    return f0


# %%
def make_f1(h_):
    n = int(round(1 / h_ - 1))
    f1 = sp.zeros(n + 1, 1)
    f1[0] = sp.integrate(f * w1rh_j.subs({h: h_, j: -1}), (x, 0, h_)).simplify()
    for j_ in range(0, n):
        f1[j_ + 1] = (
            sp.integrate(f * w1lh_j.subs({h: h_, j: j_}), (x, h_ * j_, h_ * (j_ + 1)))
            + sp.integrate(
                f * w1rh_j.subs({h: h_, j: j_}), (x, h_ * (j_ + 1), h_ * (j_ + 2))
            )
        ).simplify()
    return f1


# %%
def make_f(h_):
    n = int(round(1 / h_ - 1))
    f0 = make_f0(h_)
    f1 = make_f1(h_)
    f = sp.zeros(2 * n + 2, 1)
    for i_ in range(n + 1):
        f[i_] = f0[i_]
        f[i_ + n + 1] = f1[i_]
    return f


# %%
def make_u(h_, alpha_, f_vec):
    M = make_M(h_, alpha_)

    u = sp.Matrix(M.shape[0], 1, lambda i, j: sp.Symbol(f"u_{i + 1}"))
    f_vec_ = sp.zeros(M.shape[0], 1)
    for i in range(M.shape[0]):
        f_vec_[i] = f_vec[i]
    sol = sp.solve(M * u - f_vec_, u)

    return [sol[u[i]] for i in range(M.shape[0])]
