# %%
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import warnings

# %%
# ----- Define Numerical Basis Functions and Derivatives -----
# (These are independent of alpha and remain the same)


def wlh_func(x, h, j):
    """Numerical version of wlh = y = x/h - j"""
    return x / h - j


def wrh_func(x, h, j):
    """Numerical version of wrh = 2 - y = 2 - (x/h - j)"""
    return 2.0 + j - x / h


def wlh_p_func(x, h, j):
    """Derivative of wlh_func w.r.t x. Returns a constant 1/h."""
    return 1.0 / h


def wrh_p_func(x, h, j):
    """Derivative of wrh_func w.r.t x. Returns a constant -1/h."""
    return -1.0 / h


def wlh_func_next(x, h, j):
    """wlh function for the next interval (index j+1)"""
    return wlh_func(x, h, j + 1)


def wlh_p_func_next(x, h, j):
    """Derivative of wlh function for the next interval (index j+1)"""
    return wlh_p_func(x, h, j + 1)


# %%
# ----- Define Integrands for Quad (Simplified for 1 <= alpha < 2) -----

# --- Integrands for get_M_for_u ---
# (These did not have issues at x=0 for positive alpha, so they remain the same,
#  but we add the alpha range assumption to the description)


def integrand1_u(x, h, j, alpha):
    """Integrand for res1 in get_M_for_u. Assumes 1 <= alpha < 2."""
    # x**alpha * w0lh_j_p**2 + w0lh_j**2
    wlh_val = wlh_func(x, h, j)
    wlh_p_val = wlh_p_func(x, h, j)
    return x**alpha * wlh_p_val**2 + wlh_val**2


def integrand2_u(x, h, j, alpha):
    """Integrand for res2 and M_00_diag_first in get_M_for_u. Assumes 1 <= alpha < 2."""
    # x**alpha * w0rh_j_p**2 + w0rh_j**2
    wrh_val = wrh_func(x, h, j)
    wrh_p_val = wrh_p_func(x, h, j)
    return x**alpha * wrh_p_val**2 + wrh_val**2


def integrand_subdiag_u(x, h, j, alpha):
    """Integrand for M_00_subdiag in get_M_for_u. Assumes 1 <= alpha < 2."""
    # x**alpha * w0rh_j_p * w0lh_j_p_next + w0rh_j * w0lh_j_next
    wrh_val = wrh_func(x, h, j)
    wrh_p_val = wrh_p_func(x, h, j)
    wlh_next_val = wlh_func_next(x, h, j)
    wlh_p_next_val = wlh_p_func_next(x, h, j)
    return x**alpha * wrh_p_val * wlh_p_next_val + wrh_val * wlh_next_val


# --- Integrands for get_M_for_u_prime (SIMPLIFIED) ---
# We use the fact that 1 <= alpha < 2 implies 0 <= 2*alpha - 2 < 2.
# The term x**(2*alpha - 2) is well-behaved at x=0 (it's 1 if alpha=1, and 0 if alpha>1).
# Thus, no special handling for x=0 is needed anymore.


def integrand1_u_prime(x, h, j, alpha):
    """Integrand for res1 in get_M_for_u_prime. Assumes 1 <= alpha < 2."""
    # x**(2*alpha) * w0lh_j_p**2 + (x**alpha - alpha*(alpha-1)*x**(2*alpha-2)) * w0lh_j**2
    wlh_val = wlh_func(x, h, j)
    wlh_p_val = wlh_p_func(x, h, j)
    # This expression is now safe for 1 <= alpha < 2, even at x=0.
    # If alpha=1, term2 is alpha*(alpha-1)*... = 0. term2_coeff = x**1.
    # If alpha>1, x**(2*alpha-2) -> 0 as x->0. term2_coeff -> x**alpha.
    term2_coeff = x**alpha - alpha * (alpha - 1.0) * x ** (2.0 * alpha - 2.0)
    return x ** (2.0 * alpha) * wlh_p_val**2 + term2_coeff * wlh_val**2


def integrand2_u_prime(x, h, j, alpha):
    """Integrand for res2 and M_00_diag_first in get_M_for_u_prime. Assumes 1 <= alpha < 2."""
    # x**(2*alpha) * w0rh_j_p**2 + (x**alpha - alpha*(alpha-1)*x**(2*alpha-2)) * w0rh_j**2
    wrh_val = wrh_func(x, h, j)
    wrh_p_val = wrh_p_func(x, h, j)
    term2_coeff = x**alpha - alpha * (alpha - 1.0) * x ** (2.0 * alpha - 2.0)
    return x ** (2.0 * alpha) * wrh_p_val**2 + term2_coeff * wrh_val**2


def integrand_subdiag_u_prime(x, h, j, alpha):
    """Integrand for M_00_subdiag in get_M_for_u_prime. Assumes 1 <= alpha < 2."""
    # x**(2*alpha) * w0rh_j_p * w0lh_j_p_next + (x**alpha - alpha*(alpha-1)*x**(2*alpha-2)) * w0rh_j * w0lh_j_next
    wrh_val = wrh_func(x, h, j)
    wrh_p_val = wrh_p_func(x, h, j)
    wlh_next_val = wlh_func_next(x, h, j)
    wlh_p_next_val = wlh_p_func_next(x, h, j)
    term2_coeff = x**alpha - alpha * (alpha - 1.0) * x ** (2.0 * alpha - 2.0)
    return (
        x ** (2.0 * alpha) * wrh_p_val * wlh_p_next_val
        + term2_coeff * wrh_val * wlh_next_val
    )


# %%
# ----- Numerical Matrix Assembly Functions (with alpha assumption) -----


def get_M_for_u_numeric(h_, alpha_):
    """
    Numerically computes the matrix M for u using scipy.integrate.quad.
    Assumes 1 <= alpha < 2.

    Args:
        h_ (float): Element size (0 < h_ <= 1).
        alpha_ (float): Exponent parameter (must be >= 1 and < 2).

    Returns:
        numpy.ndarray: The computed matrix M_00.
    """
    if not (1.0 <= alpha_ < 2.0):
        raise ValueError(f"alpha_ must be in the range [1, 2), but got {alpha_}")
    if not (0 < h_ <= 1.0):
        raise ValueError(f"h_ must be in the range (0, 1], but got {h_}")

    # Calculate matrix size N+1 where N = 1/h_ - 1
    # Need to handle potential floating point inaccuracies in 1/h_
    n_float = 1.0 / h_ - 1.0
    if np.isclose(n_float, round(n_float)):
        n = int(round(n_float))
    else:
        # This case should ideally not happen if h_ is a divisor of 1, e.g. 0.2, 0.1 etc.
        warnings.warn(
            f"h_={h_} does not seem to be an integer fraction of 1. Resulting matrix size might be unexpected.",
            UserWarning,
        )
        n = int(np.floor(n_float))  # Or ceiling, depending on interpretation

    matrix_size = n + 1
    if matrix_size < 1:
        raise ValueError(
            f"h_={h_} leads to invalid matrix size {matrix_size}. Check if h_ > 1."
        )

    M_00 = np.zeros((matrix_size, matrix_size))

    # Integration parameters for quad
    quad_opts = {"limit": 100, "epsabs": 1e-9, "epsrel": 1e-9}

    # Loop through nodes (j_ corresponds to the 'j' in the symbolic code's basis definitions)
    # Row/Col indices k go from 0 to n. Basis function phi_k is centered at x=kh.
    # phi_k involves wlh(j=k-1) and wrh(j=k-1).
    # The symbolic loop index 'j_' is equivalent to 'k-1'.
    for k in range(matrix_size):  # k = 0, 1, ..., n
        j_ = k - 1  # Corresponding 'j' index for basis centered at k

        # --- Calculate Diagonal Elements M_00[k, k] ---
        # M[k, k] = Integral( L(phi_k) * phi_k ) over support of phi_k
        #         = Integral over [(k-1)h, kh] + Integral over [kh, (k+1)h]
        # The symbolic code's res1+res2 directly calculates this diagonal using index j_.

        # Integral 1: uses wlh_j -> integrand1_u(j_) over [j_h, (j_+1)h] = [(k-1)h, kh]
        lower1 = h_ * j_
        upper1 = h_ * (j_ + 1)
        if k == 0:  # phi_0 only lives on [0, h]. No left interval.
            res1_integral = 0.0
        else:
            val, abserr = quad(
                integrand1_u, lower1, upper1, args=(h_, j_, alpha_), **quad_opts
            )
            # Optional: Check abserr against tolerance
            if abserr > 1e-6:
                warnings.warn(
                    f"High integration error ({abserr:.2e}) in M_u diag k={k} (res1)",
                    UserWarning,
                )
            res1_integral = val

        # Integral 2: uses wrh_j -> integrand2_u(j_) over [(j_+1)h, (j_+2)h] = [kh, (k+1)h]
        lower2 = h_ * (j_ + 1)
        upper2 = h_ * (j_ + 2)
        # Ensure upper integration limit doesn't exceed domain [0, 1]
        upper2 = min(upper2, 1.0)
        # Can happen if k=n, then (k+1)h = (n+1)h = 1
        if lower2 >= upper2:  # e.g. if k=n and h is such that (n+1)h=1 exactly
            res2_integral = 0.0
        else:
            val, abserr = quad(
                integrand2_u, lower2, upper2, args=(h_, j_, alpha_), **quad_opts
            )
            if abserr > 1e-6:
                warnings.warn(
                    f"High integration error ({abserr:.2e}) in M_u diag k={k} (res2)",
                    UserWarning,
                )
            res2_integral = val

        M_00[k, k] = res1_integral + res2_integral

        # --- Calculate Off-Diagonal Elements M_00[k, k+1] ---
        # Based on symbolic code: M[row, row-1] uses j_sub = j_ - 1 = k - 2
        # We calculate M[k, k+1] which involves phi_k and phi_{k+1}
        # Overlap is on interval [kh, (k+1)h]
        # On this interval: phi_k uses wrh(j=k-1) and phi_{k+1} uses wlh(j=k)
        # The symbolic subdiagonal integral uses wrh_j * wlh_{j+1} over interval [(j+1)h, (j+2)h]
        # If we set the symbolic 'j' to our 'j_' (which is k-1), the interval is [kh, (k+1)h]
        # The integrand becomes wrh(j=k-1) * wlh(j=k), which matches our requirement.
        # So we use integrand_subdiag_u with j_ = k-1

        if k < n:  # Check if there is a next element k+1
            j_for_subdiag = j_  # = k-1
            lower_sub = h_ * (j_for_subdiag + 1)  # = kh
            upper_sub = h_ * (j_for_subdiag + 2)  # = (k+1)h
            # Ensure upper limit doesn't exceed 1.0
            upper_sub = min(upper_sub, 1.0)

            if lower_sub < upper_sub:
                val, abserr = quad(
                    integrand_subdiag_u,
                    lower_sub,
                    upper_sub,
                    args=(h_, j_for_subdiag, alpha_),
                    **quad_opts,
                )
                if abserr > 1e-6:
                    warnings.warn(
                        f"High integration error ({abserr:.2e}) in M_u subdiag k={k}, k+1",
                        UserWarning,
                    )

                M_00[k, k + 1] = val
                M_00[k + 1, k] = val  # Symmetric matrix

    return M_00


def get_M_for_u_prime_numeric(h_, alpha_):
    """
    Numerically computes the matrix M for u' using scipy.integrate.quad.
    Assumes 1 <= alpha < 2. Integrands are simplified based on this.

    Args:
        h_ (float): Element size (0 < h_ <= 1).
        alpha_ (float): Exponent parameter (must be >= 1 and < 2).

    Returns:
        numpy.ndarray: The computed matrix M_00.
    """
    if not (1.0 <= alpha_ < 2.0):
        raise ValueError(f"alpha_ must be in the range [1, 2), but got {alpha_}")
    if not (0 < h_ <= 1.0):
        raise ValueError(f"h_ must be in the range (0, 1], but got {h_}")

    # Calculate matrix size N+1 where N = 1/h_ - 1
    n_float = 1.0 / h_ - 1.0
    if np.isclose(n_float, round(n_float)):
        n = int(round(n_float))
    else:
        warnings.warn(
            f"h_={h_} does not seem to be an integer fraction of 1. Resulting matrix size might be unexpected.",
            UserWarning,
        )
        n = int(np.floor(n_float))

    matrix_size = n + 1
    if matrix_size < 1:
        raise ValueError(
            f"h_={h_} leads to invalid matrix size {matrix_size}. Check if h_ > 1."
        )

    M_00 = np.zeros((matrix_size, matrix_size))

    # Integration parameters for quad
    quad_opts = {"limit": 100, "epsabs": 1e-9, "epsrel": 1e-9}

    # Loop logic identical to get_M_for_u_numeric, just using different integrands
    for k in range(matrix_size):  # k = 0, 1, ..., n
        j_ = k - 1  # Corresponding 'j' index for basis centered at k

        # --- Calculate Diagonal Elements M_00[k, k] ---
        # Integral 1: uses integrand1_u_prime(j_) over [(k-1)h, kh]
        lower1 = h_ * j_
        upper1 = h_ * (j_ + 1)
        if k == 0:
            res1_integral = 0.0
        else:
            val, abserr = quad(
                integrand1_u_prime, lower1, upper1, args=(h_, j_, alpha_), **quad_opts
            )
            if abserr > 1e-6:
                warnings.warn(
                    f"High integration error ({abserr:.2e}) in M_u_prime diag k={k} (res1)",
                    UserWarning,
                )
            res1_integral = val

        # Integral 2: uses integrand2_u_prime(j_) over [kh, (k+1)h]
        lower2 = h_ * (j_ + 1)
        upper2 = h_ * (j_ + 2)
        upper2 = min(upper2, 1.0)
        if lower2 >= upper2:
            res2_integral = 0.0
        else:
            val, abserr = quad(
                integrand2_u_prime, lower2, upper2, args=(h_, j_, alpha_), **quad_opts
            )
            if abserr > 1e-6:
                warnings.warn(
                    f"High integration error ({abserr:.2e}) in M_u_prime diag k={k} (res2)",
                    UserWarning,
                )
            res2_integral = val

        M_00[k, k] = res1_integral + res2_integral

        # --- Calculate Off-Diagonal Elements M_00[k, k+1] ---
        # Uses integrand_subdiag_u_prime with j_ = k-1 over [kh, (k+1)h]
        if k < n:
            j_for_subdiag = j_  # = k-1
            lower_sub = h_ * (j_for_subdiag + 1)  # = kh
            upper_sub = h_ * (j_for_subdiag + 2)  # = (k+1)h
            upper_sub = min(upper_sub, 1.0)

            if lower_sub < upper_sub:
                val, abserr = quad(
                    integrand_subdiag_u_prime,
                    lower_sub,
                    upper_sub,
                    args=(h_, j_for_subdiag, alpha_),
                    **quad_opts,
                )
                if abserr > 1e-6:
                    warnings.warn(
                        f"High integration error ({abserr:.2e}) in M_u_prime subdiag k={k}, k+1",
                        UserWarning,
                    )

                M_00[k, k + 1] = val
                M_00[k + 1, k] = val  # Symmetric matrix

    return M_00


# %%
# ----- Example Usage -----
if __name__ == "__main__":
    h_val = 0.2  # Example element size
    alpha_val = 1.5  # Example alpha value in [1, 2)

    print(f"Calculating M_for_u with h={h_val}, alpha={alpha_val}")
    try:
        M_u_numeric = get_M_for_u_numeric(h_val, alpha_val)
        print("M_for_u (Numeric):\n", M_u_numeric)
    except ValueError as e:
        print("Error:", e)
    print("-" * 30)

    # Test boundary case alpha=1.0
    alpha_val_1 = 1.0
    print(f"Calculating M_for_u with h={h_val}, alpha={alpha_val_1}")
    try:
        M_u_numeric_a1 = get_M_for_u_numeric(h_val, alpha_val_1)
        print("M_for_u (Numeric, alpha=1.0):\n", M_u_numeric_a1)
    except ValueError as e:
        print("Error:", e)
    print("-" * 30)

    print(f"Calculating M_for_u_prime with h={h_val}, alpha={alpha_val}")
    try:
        M_u_prime_numeric = get_M_for_u_prime_numeric(h_val, alpha_val)
        print("M_for_u_prime (Numeric):\n", M_u_prime_numeric)
    except ValueError as e:
        print("Error:", e)
    print("-" * 30)

    # Test boundary case alpha=1.0 for u_prime
    print(f"Calculating M_for_u_prime with h={h_val}, alpha={alpha_val_1}")
    try:
        M_u_prime_numeric_a1 = get_M_for_u_prime_numeric(h_val, alpha_val_1)
        print("M_for_u_prime (Numeric, alpha=1.0):\n", M_u_prime_numeric_a1)
    except ValueError as e:
        print("Error", e)
    print("-" * 30)

    # Test invalid alpha
    alpha_invalid = 0.7
    print(f"Testing invalid alpha={alpha_invalid}")
    try:
        get_M_for_u_numeric(h_val, alpha_invalid)
    except ValueError as e:
        print("Caught expected error for M_u:", e)
    try:
        get_M_for_u_prime_numeric(h_val, alpha_invalid)
    except ValueError as e:
        print("Caught expected error for M_u_prime:", e)
    print("-" * 30)
