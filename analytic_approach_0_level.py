# %%
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    for j_ in tqdm(range(-1, n), desc=f"Building M_00 (h={h_})", leave=False):  # <-- tqdm added
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
def make_M(h_):
    return make_M_00(h_)


# %%
f = -((-1 + x) ** 3) + 3 * (-1 + x) * x ** (-1 + alpha) * (-alpha + x * (2 + alpha))
f = (x ** (3 - alpha) - 2 * (3 - alpha) * x - 1) / (3 - alpha)
f = 3 * (x - 1) * x ** (alpha - 1) * ((alpha + 2) * x - alpha) - (x - 1) ** 3
f = (x**(3-alpha)-2*(3-alpha)*x-1)/(3-alpha)

# %%
def make_f0(h_, alpha_=None):
    n = int(round(1 / h_ - 1))
    f0 = sp.zeros(n + 1, 1)
    f0[0] = sp.integrate(f.subs({alpha: alpha_}) * w0rh_j.subs({h: h_, j: -1}), (x, 0, h_)).simplify()
    for j_ in tqdm(range(0, n), desc=f"Building f0 (h={h_})", leave=False):  # <-- tqdm added
        f0[j_ + 1] = (
            sp.integrate(f.subs({alpha: alpha_}) * w0lh_j.subs({h: h_, j: j_}), (x, h_ * j_, h_ * (j_ + 1)))
            + sp.integrate(
                f.subs({alpha: alpha_}) * w0rh_j.subs({h: h_, j: j_}), (x, h_ * (j_ + 1), h_ * (j_ + 2))
            )
        ).simplify()
    # if alpha_ is not None:
    #     return f0.subs({alpha: alpha_})
    return f0


# %%
def make_f(h_, alpha_=None):
    return make_f0(h_, alpha_)


# %%
def make_u(h_, alpha_):
    f_vec = make_f(h_, alpha_)
    print(f"{f_vec=}")
    M = make_M(h_).subs({alpha: alpha_})

    u = sp.Matrix(M.shape[0], 1, lambda i, j: sp.Symbol(f"u_{i + 1}"))
    sol = sp.solve(M * u - f_vec, u)

    return [sol[u[i]] for i in range(M.shape[0])]


# %%
# make_f(0.25).subs({alpha: 1})
# %%

# Convert to numerical functions
h_val = sp.S(1)/10000
j_val = 1
alpha_val = 1  # Not used in these functions

w0lh_j_num = sp.lambdify(x, w0lh_j.subs({h: h_val, j: j_val}), 'numpy')
w0rh_j_num = sp.lambdify(x, w0rh_j.subs({h: h_val, j: j_val}), 'numpy')

# Create x values for plotting
x_vals_lh = np.linspace(j_val * h_val, (j_val + 1) * h_val, 100)
x_vals_rh = np.linspace((j_val+1) * h_val, (j_val + 2) * h_val, 100)

# Evaluate functions
y_lh = w0lh_j_num(x_vals_lh)
y_rh = w0rh_j_num(x_vals_rh)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals_lh, y_lh, label='w0lh_j')
plt.plot(x_vals_rh, y_rh, label='w0rh_j')
plt.axvline(x=j_val*h_val, color='gray', linestyle='--')
plt.axvline(x=(j_val+1)*h_val, color='gray', linestyle='--')
plt.title(f'Basis functions with h={h_val}, j={j_val}')
plt.xlabel('x')
plt.ylabel('Function value')
plt.legend()
plt.grid(True)
plt.show()

# %%
# A0 = make_u(h_val,alpha_val)
M = make_M(h_val).subs({alpha: alpha_val})
f_vec = make_f(h_val, alpha_val)
#%%
def solve_tridiagonal_system(M, f_vec, show_progress=True):
    """
    Solves a tridiagonal system M·u = f_vec using the Thomas algorithm.
    
    Parameters:
    - M: A sympy tridiagonal matrix
    - f_vec: A sympy vector (Matrix with one column)
    - show_progress: Whether to show progress bars (default: True)
    
    Returns:
    - u: Solution vector
    """
    n = len(f_vec)
    
    # Extract the diagonals from M
    a = [0]  # subdiagonal (below main diagonal)
    b = []   # main diagonal
    c = []   # superdiagonal (above main diagonal)
    
    # Extract the diagonals
    extraction_iter = range(n)
    if show_progress:
        extraction_iter = tqdm(extraction_iter, desc="Extracting diagonals")
    
    for i in extraction_iter:
        b.append(M[i, i])
        if i < n-1:
            c.append(M[i, i+1])
        if i > 0:
            a.append(M[i, i-1])
    
    # Convert f_vec to a list
    d = [f_vec[i] for i in range(n)]
    
    # Forward elimination
    forward_iter = range(1, n)
    if show_progress:
        forward_iter = tqdm(forward_iter, desc="Forward elimination")
    
    for i in forward_iter:
        w = a[i] / b[i-1]
        b[i] = b[i] - w * c[i-1]
        d[i] = d[i] - w * d[i-1]
    
    # Back substitution
    x = [0] * n
    x[n-1] = d[n-1] / b[n-1]
    
    back_iter = range(n-2, -1, -1)
    if show_progress:
        back_iter = tqdm(back_iter, desc="Back substitution")
    
    for i in back_iter:
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    # Return the solution as a sympy Matrix
    return x
#%%
def make_u2(m00, f0):
    n = len(f0)
    
    # Extract the diagonals from M
    a = [np.longdouble(0)]  # subdiagonal (below main diagonal)
    b = []   # main diagonal
    c = []   # superdiagonal (above main diagonal)
    
    # Extract the diagonals
    extraction_iter = range(n)
    extraction_iter = tqdm(extraction_iter, desc="Extracting diagonals")
    
    def transform(v):
        return np.longdouble(v.p) / np.longdouble(v.q)

    for i in extraction_iter:
        b.append(transform(M[i, i]))
        if i < n-1:
            c.append(transform(M[i, i+1]))
        if i > 0:
            a.append(transform(M[i, i-1]))

    d = [transform(f0[i]) for i in range(n)]

    # Forward elimination
    forward_iter = range(1, n)
    forward_iter = tqdm(forward_iter, desc="Forward elimination")
    
    for i in forward_iter:
        w = a[i] / b[i-1]
        b[i] = b[i] - w * c[i-1]
        d[i] = d[i] - w * d[i-1]
    
    # Back substitution
    x = [0] * n
    x[n-1] = d[n-1] / b[n-1]
    
    back_iter = range(n-2, -1, -1)
    back_iter = tqdm(back_iter, desc="Back substitution")
    
    for i in back_iter:
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    # Return the solution as a sympy Matrix
    return x
#%%
u = solve_tridiagonal_system(M, f_vec)
u
#%%
u = make_u2(M, f_vec)
u
# %%# Parameters
alpha = alpha_val
h = h_val
n = int(round(1 / h - 1))
# %%
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
    result[mask1] = y[mask1]

    # Region 1 < y <= 2
    mask2 = (y > 1) & (y <= 2)
    result[mask2] = 2 - y[mask2]

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




# %%
phi0Compiled = [make_phi0(j) for j in tqdm(range(-1, n), desc="Generating phi0 functions")]  # <-- tqdm added
A0 = u
# %%
def u_approx(x):
    result = 0
    k = round(x / h_val)
    from_idx = max(0, k - 2)
    to_idx = min(len(A0) - 1, k + 2)
    for idx in tqdm(range(from_idx, to_idx), desc="Computing u_approx", leave=False):  # <-- tqdm added
        result += A0[idx] * phi0Compiled[idx](x)
    return result

u_approx_vec = np.vectorize(u_approx)

def uexact_func(x, alpha):
    return (x ** (3 - alpha) - 1) / (3 - alpha)

# %%
x_vals = np.linspace(0, 1, 500)
u_approx_vec = np.vectorize(u_approx)
plt.plot(x_vals, u_approx_vec(x_vals), label="Approximate solution $u_{approx}(x)$")
plt.plot(
    x_vals,
    uexact_func(x_vals, alpha),
    label="Exact solution $u_{exact}(x)$",
    linestyle="--",
)
plt.scatter(
    [float(h * j) for j in range(len(A0))],
    A0,
    alpha=0.2,
)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Approximate vs Exact Solution")
plt.legend()
plt.show()

# %%
plt.plot(x_vals, np.abs(uexact_func(x_vals, alpha)-u_approx_vec(x_vals)), label="Err Approximate solution $u_{approx}(x)$")
plt.xlabel("x")
plt.ylabel("delta(x)")
plt.title("Error of Solution")
plt.legend()
plt.show()
# %%

print('\n'.join(f'[{np.abs(uexact_func(x,alpha)-u_approx(x)):.2e}],' for x in np.linspace(0,0.9,10)))
# %%
def print_typst_data(xs, ys, label):
    label = label.replace(" ", "_").lower()
    print(f"#let {label} = ({', '.join(f'({x}, {y})' for x, y in zip(xs, ys))})")
# %%
print_typst_data(x_vals, np.abs(uexact_func(x_vals, alpha)-u_approx_vec(x_vals)), "error_data")
