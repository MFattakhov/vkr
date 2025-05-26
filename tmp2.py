#%%
from sympy import symbols, Function, diff, simplify, exp, factor, expand, cancel, collect, pprint
#%%
# --- 1. Define symbols and transformation ---
v, x = symbols('v x', real=True)
kappa, theta, sigma = symbols('kappa theta sigma', real=True, positive=True)

px_func = Function('px_func')(x) # This is P(x) from Gardiner, where P(x)dx = pv(v)dv
pv_func = Function('pv_func')(v) # Original p_v(v)

# Transformation: v = x/(1-x)  => x = v/(1+v)
v_of_x = x / (1 - x)
x_of_v = v / (1 + v)

# Derivatives of transformation
# dv_dx = 1/(1-x)^2
# dx_dv = 1/(1+v)^2
dv_dx_expr = diff(v_of_x, x) # This is dv/dx as a function of x
dx_dv_expr = diff(x_of_v, v) # This is dx/dv as a function of v

print(f"dv_dx = {simplify(dv_dx_expr)}")
print(f"dx_dv = {simplify(dx_dv_expr)}")

# CORRECTED Relationship: pv(v) = px_func(x(v)) * dx/dv
# Alternatively, pv(v(x)) = px_func(x) * dx/dv.subs(v, v_of_x)
# Or, to substitute into the ODE in v:
# pv_func(v) corresponds to px_func(x(v)) * dx_dv_expr (where dx_dv_expr is a function of v)
pv_substituted = px_func.subs(x, x_of_v) * dx_dv_expr

# --- 2. Derivatives of pv_substituted w.r.t v in terms of px_func(x) and its derivatives w.r.t x ---
pv_prime_v = diff(pv_substituted, v)
pv_double_prime_v = diff(pv_substituted, v, 2) # Corrected to diff pv_substituted, not pv_prime_v further

# --- 3. Substitute into ODE for pv_func(v) ---
# Original ODE: A0_v_coeff * pv_func'' + A1_v_coeff * pv_func' + A2_v_coeff * pv_func = 0
A0_v_coeff_sym = (sigma**2 * v) / 2
A1_v_coeff_sym = (sigma**2 - kappa*theta + kappa*v)
A2_v_coeff_sym = kappa

ode_orig_pv = A0_v_coeff_sym * pv_double_prime_v + \
              A1_v_coeff_sym * pv_prime_v + \
              A2_v_coeff_sym * pv_substituted # Corrected to pv_substituted

# Now, substitute v = v_of_x to get everything in terms of x
ode_in_x = simplify(ode_orig_pv.subs(v, v_of_x))

# --- 4. Identify coefficients A0_x, A1_x, A2_x for px_func(x) ---
# The functions in ode_in_x are px_func(x), Derivative(px_func(x),x), Derivative(px_func(x),x,x)
px_d = px_func.diff(x)
px_dd = px_func.diff(x,2)

# Expand to make collection easier (sometimes simplify does this, but being explicit)
ode_in_x_expanded = expand(ode_in_x)

A0_x_coeff = simplify(ode_in_x_expanded.coeff(px_dd))
A1_x_coeff = simplify(ode_in_x_expanded.coeff(px_d))
# For A2_x, collect first then get coeff of px_func(x)
temp_collected = collect(ode_in_x_expanded, [px_dd, px_d])
A2_x_coeff = simplify(temp_collected.coeff(px_func(x)))


print("\n--- Coefficients of the ODE for px_func(x): A0_x px_func'' + A1_x px_func' + A2_x px_func = 0 ---")
print("A0_x(x) = ")
pprint(A0_x_coeff)
# Expected D_x(x) from Gardiner: (1/2)*sigma**2*x*(1-x)**3
D_x_expected = (sigma**2 * x * (1-x)**3) / 2
print("Check against D_x_expected from Gardiner:")
pprint(simplify(A0_x_coeff - D_x_expected))


# Coefficients relevant for Gardiner's form: $0 = -d/dx [μ_x(x) P(x)] + d^2/dx^2 [D_x(x) P(x)]$
# D_x(x) = A0_x_coeff (coefficient of P'')
# mu_x_gardiner(x) = A1_x_coeff - 2 * diff(A0_x_coeff, x) # From expanding (D_x P)'' - (mu_x P)'
# No, this is not right.
# The expanded form is D_x P'' + (2D_x' - mu_x_gardiner)P' + (D_x'' - mu_x_gardiner')P = 0
# So:
# A0_x = D_x
# A1_x = 2D_x' - mu_x_gardiner  => mu_x_gardiner = 2D_x' - A1_x
# A2_x = D_x'' - mu_x_gardiner'

# The integrand for the exponential part of the Sturm-Liouville integrating factor is:
# (A1_x - A0_x') / A0_x = ( (2D_x' - mu_x_gardiner) - D_x' ) / D_x = (D_x' - mu_x_gardiner) / D_x
# This is what we expect for ln(mu_P_integrating_factor) = ln(D_x) - integral(mu_x_gardiner/D_x)

A0_x_prime = simplify(diff(A0_x_coeff, x))
log_mu_integrand = simplify( (A1_x_coeff - A0_x_prime) / A0_x_coeff )

print("\nIntegrand for log(mu_SturmLiouville_factor(x)): (A1_x - A0_x') / A0_x = ")
pprint(log_mu_integrand)

# This is (D_x' - mu_x_gardiner) / D_x
# where mu_x_gardiner = 2*A0_x_prime - A1_x_coeff

mu_x_gardiner_derived_from_coeffs = simplify(2*A0_x_prime - A1_x_coeff)
integrand_for_exp_term_in_SL_factor = simplify(mu_x_gardiner_derived_from_coeffs / A0_x_coeff) # This is mu_x_gardiner / D_x

print("\nIntegrand for exp term in Sturm-Liouville mu_SL_factor, corresponds to (mu_x_gardiner / D_x):")
pprint(integrand_for_exp_term_in_SL_factor)

# Expected manual integrand for (mu_x_gardiner / D_x)
term1_manual_integrand = (2*kappa/sigma**2) * (theta-(theta+1)*x) / (x * (1-x)**2)
term2_manual_integrand = (2*x-1) / (x*(1-x))
expected_manual_integrand = simplify(term1_manual_integrand + term2_manual_integrand)
print("\nExpected integrand for (mu_x_gardiner / D_x) from manual calc:")
pprint(expected_manual_integrand)

print("\nDifference between SymPy derived (mu_x_gardiner/D_x) and manual one:")
diff_integrands = simplify(integrand_for_exp_term_in_SL_factor - expected_manual_integrand)
pprint(diff_integrands)
if diff_integrands == 0:
    print("Integrands for (mu_x_gardiner/D_x) match!")
else:
    print("Integrands for (mu_x_gardiner/D_x) DO NOT match. Check derivations.")

print("\nA1_x(x) = ")
pprint(A1_x_coeff)
print("\nA2_x(x) = ")
pprint(A2_x_coeff)


# %%
