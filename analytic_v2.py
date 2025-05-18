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


def get_M_for_u(h_, alpha_):
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
        sp.integrate(x**alpha * w0rh_j_p**2 + w0rh_j**2, (x, 0, h))
        .subs({j: -1})
        .simplify()
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

    return make_M_00(h_).subs({alpha: alpha_})


def get_M_for_u_prime(h_, alpha_):
    # integrate
    res1 = sp.integrate(
        (
            x ** (2 * alpha) * w0lh_j_p**2
            + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2)) * w0lh_j**2
        ).subs({alpha: alpha_}),
        (x, h * j, h * (j + 1)),
    ).simplify()
    res2 = sp.integrate(
        (
            x ** (2 * alpha) * w0rh_j_p**2
            + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2)) * w0rh_j**2
        ).subs({alpha: alpha_}),
        (x, h * (j + 1), h * (j + 2)),
    ).simplify()
    M_00_diag_inner = (res1 + res2).simplify()

    M_00_diag_first = (
        sp.integrate(
            (
                x ** (2 * alpha) * w0rh_j_p**2
                + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2)) * w0rh_j**2
            ).subs({alpha: alpha_}),
            (x, sp.S(1) / 2**16, h),
        )
        .subs({j: -1})
        .simplify()
    )

    M_00_subdiag_first = sp.integrate(
        (
            x ** (2 * alpha) * w0rh_j_p * w0lh_j_p_next
            + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2))
            * w0rh_j
            * w0lh_j_next
        ).subs({alpha: alpha_}),
        (x, sp.S(1) / 2**16, h),
    ).subs({j: -1})
    M_00_subdiag = sp.integrate(
        (
            x ** (2 * alpha) * w0rh_j_p * w0lh_j_p_next
            + (x**alpha - alpha * (alpha - 1) * x ** (2 * alpha - 2))
            * w0rh_j
            * w0lh_j_next
        ).subs({alpha: alpha_}),
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
                if row != 1:
                    M_00[row, row - 1] = M_00_subdiag.subs({h: h_, j: j_ - 1})
                    M_00[row - 1, row] = M_00_subdiag.subs({h: h_, j: j_ - 1})
                else:
                    M_00[row, row - 1] = M_00_subdiag_first.subs({h: h_})
                    M_00[row - 1, row] = M_00_subdiag_first.subs({h: h_})
        return M_00

    return make_M_00(h_)
