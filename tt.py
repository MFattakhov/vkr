#%%
from sympy import symbols, Function, diff, exp, simplify, Eq, pprint, S
#%%
# Define symbols and the function p(v)
v = symbols('v', real=True, positive=True) # Assume v > 0 (e.g., variance)
kappa, theta, sigma = symbols('kappa theta sigma', real=True, positive=True) # sigma > 0
p = Function('p')(v)

# Derivatives of p(v)
p_prime = p.diff(v)
p_double_prime = p.diff(v, 2)

# --- 1. Original Equation ---
# Compact form: 0 = - d/dv [(κ(θ - v))p] + 1/2 d²/dv² [σ² v p]
term1_orig_compact = -diff( (kappa*(theta-v)) * p, v)
term2_orig_compact_inner = sigma**2 * v * p
term2_orig_compact = S(1)/2 * diff(term2_orig_compact_inner, v, 2)
original_lhs_compact = term1_orig_compact + term2_orig_compact

# Expanded form: (1/2 σ²v)p'' + (σ² - κθ + κv)p' + κp = 0
A0 = (sigma**2 * v) / 2
A1 = (sigma**2 - kappa*theta + kappa*v)
A2 = kappa
original_lhs_expanded = A0*p_double_prime + A1*p_prime + A2*p

# Verify that our manual expansion of the original equation is correct
print("Verifying manual expansion of the original equation:")
expansion_check = simplify(original_lhs_compact.expand() - original_lhs_expanded.expand())
print(f"Compact form expanded - Manual expanded form = {expansion_check}")
if expansion_check == 0:
    print("Manual expansion is correct.")
else:
    print("Error in manual expansion!")
    # pprint(original_lhs_compact.expand())
    # pprint(original_lhs_expanded)

print("\nOriginal LHS (from manual expansion):")
pprint(original_lhs_expanded)

# --- 2. Derived Parameters for the Target Form ---
alpha_val = 2 - (2*kappa*theta)/(sigma**2)
s_v = (sigma**2 / 2) * exp((2*kappa*v)/(sigma**2))
q_v = -kappa * v**(1 - (2*kappa*theta)/(sigma**2)) * exp((2*kappa*v)/(sigma**2))
f_v = 0 # Given by the problem

print("\nDerived parameters:")
print(f"alpha = {alpha_val}")
print(f"s(v) =")
pprint(s_v)
print(f"q(v) =")
pprint(q_v)
print(f"f(v) = {f_v}")

# --- 3. Transformed Equation LHS: -(v^alpha s(v) p'(v))' + q(v) p(v) ---
S_full = v**alpha_val * s_v
term1_transformed = -diff(S_full * p_prime, v)
term2_transformed = q_v * p
transformed_lhs = term1_transformed + term2_transformed

print("\nTransformed LHS (M(p)):")
# We need to expand it to compare coefficients later if needed
transformed_lhs_expanded = transformed_lhs.expand()
pprint(transformed_lhs_expanded)

# --- 4. Integrating Factor ---
# mu(v) = C * v^(1 - 2κθ/σ²) * e^(2κv/σ²)
# We chose C = -1 for s(v) and q(v) definitions.
# S_full(v) = -μ(v)A_0(v)  => v^alpha s(v) = -μ(v) (1/2 σ²v)
#  v^(2 - 2κθ/σ²) * (σ²/2) e^(2κv/σ²) = -μ(v) (1/2 σ²v)
#  v^(1 - 2κθ/σ²) * e^(2κv/σ²) = -μ(v)
# So, mu(v) = - v^(1 - 2κθ/σ²) * e^(2κv/σ²)
mu_v_exponent_v = 1 - (2*kappa*theta)/(sigma**2)
mu_v_exponent_exp = (2*kappa*v)/(sigma**2)
mu_v = - (v**mu_v_exponent_v * exp(mu_v_exponent_exp))

print(f"\nIntegrating factor mu(v):")
pprint(mu_v)

# --- 5. Verification: Check if M(p) = μ(v) * L(p) ---
# M(p) is transformed_lhs
# L(p) is original_lhs_expanded
# We want to check if transformed_lhs - mu_v * original_lhs_expanded == 0

difference = transformed_lhs_expanded - (mu_v * original_lhs_expanded).expand()
simplified_difference = simplify(difference)

print("\nDifference (Transformed LHS - mu*Original LHS):")
pprint(simplified_difference)

if simplified_difference == 0:
    print("\n--- VERIFICATION SUCCESSFUL ---")
    print("The transformed equation form matches mu(v) times the original expanded equation.")
else:
    print("\n--- VERIFICATION FAILED ---")
    print("The expressions do not simplify to zero.")
    print("Transformed LHS coefficients:")
    # Collect coefficients of p, p', p'' for transformed_lhs_expanded
    c_p_transformed = simplify(transformed_lhs_expanded.coeff(p))
    c_p_prime_transformed = simplify(transformed_lhs_expanded.coeff(p_prime))
    c_p_double_prime_transformed = simplify(transformed_lhs_expanded.coeff(p_double_prime))
    # pprint(f"Coeff p: {c_p_transformed}")
    # pprint(f"Coeff p': {c_p_prime_transformed}")
    # pprint(f"Coeff p'': {c_p_double_prime_transformed}")

    print("\n(mu * Original LHS) coefficients:")
    mu_orig_expanded = (mu_v * original_lhs_expanded).expand()
    c_p_mu_orig = simplify(mu_orig_expanded.coeff(p))
    c_p_prime_mu_orig = simplify(mu_orig_expanded.coeff(p_prime))
    c_p_double_prime_mu_orig = simplify(mu_orig_expanded.coeff(p_double_prime))
    # pprint(f"Coeff p: {c_p_mu_orig}")
    # pprint(f"Coeff p': {c_p_prime_mu_orig}")
    # pprint(f"Coeff p'': {c_p_double_prime_mu_orig}")

    # To be more explicit, let's check each coefficient:
    print("\nCoefficient comparison:")
    print(f"p term matches: {simplify(c_p_transformed - c_p_mu_orig) == 0}")
    print(f"p' term matches: {simplify(c_p_prime_transformed - c_p_prime_mu_orig) == 0}")
    print(f"p'' term matches: {simplify(c_p_double_prime_transformed - c_p_double_prime_mu_orig) == 0}")


# %%
transformed_lhs_expanded
# %%
mu_v
# %%
transformed_lhs_expanded
# %%
alpha_val
# %%
s_v
# %%
transformed_lhs
# %%
q_v
# %%
f_v
# %%
