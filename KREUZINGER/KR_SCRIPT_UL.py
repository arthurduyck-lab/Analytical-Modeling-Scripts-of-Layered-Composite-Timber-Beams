# ===============================================================
# FUNCTIONS & IMPORTS
# ===============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from KR_FUNCTIONS import (read_data, KR_Rigidity, KR_Parameters,
                          KR_A_Sigma, KR_A_Tau, KR_B_Sigma, KR_B_Tau,
                          KR_sigma_max_per_lamella, KR_tau_max_per_lamella,
                          sum_data, FS_q, FS_2q, KR_Ts, KR_ConnectorForces,
                          KR_dMdx)

# ===============================================================
# INPUT DATA
# ===============================================================

## NAME CONFIG
name = 'KR_UL'

## LAMELLAE PARAMETERS
L_beam = 1400.0    # Beam length [mm]

## LOAD PARAMETERS
qm = 1                # [N/mm]

## SECTION FOR STRESSES
x0_sigma = L_beam/2     # Location for bending stress
x0_tau = 0.0            # Location for shear stress

## INTERMEDIATES PARAMETERS
# Lamellae
omega = np.pi / float(L_beam)

path_LAM = 'data_LAM.csv'      # tab-separated text file
path_CON = 'data_CON.csv'      # tab-separated text file
LAM = read_data(path_LAM, sep='\t')
CON = read_data(path_CON, sep='\t')


# PRINT DATA
print(f'============================================ {name} ==============================================')
print(f'Forces : qm = {qm} [N/mm]')
print(f'Length of the beam : L = {L_beam} [mm]')
print('\n')

print(f'Lamellae properties')
print(LAM)
print('\n')

print(f'Connector properties properties')
print(CON)
print('\n')

# DATA CONSISTENCY CHECK
n_LAM = len(LAM)
n_CON = len(CON)

if n_CON != (n_LAM - 1):
    print('ERROR - Incomplete connectors or lamellae data')
else:
    print('OK - Valid number of input data')


# ===============================================================
# DISCRETIZATION
# ===============================================================
n_points = 1001
x_vals = np.linspace(0, L_beam, n_points)


# ===============================================================
# APPLIED LOAD
# ===============================================================
def p_expr(x):
    """Uniform load"""
    return np.full_like(np.atleast_1d(x), qm, dtype=float)


# ===============================================================
# STEP 0 — COMPUTE BEAM PARAMETERS AND RIGIDITIES
# ===============================================================
h, b, E, G, EA, EI, n, z0, zG, z, a, s = KR_Parameters(LAM, CON)
alpha_val, GSef_val, EIa_val, EIb_val = KR_Rigidity(LAM, CON)

print("Step 0: alpha, GSef, EIa, EIb --- OK")


# ===============================================================
# STEP 1 — SOLVE: p_k'' - α*p_k = -(GSef/EIa)*p(x)
# Boundary conditions: p_k(0)=0, p_k(L)=0
# ===============================================================
def ode_pk(x, y):
    # y[0] = p_k, y[1] = p_k'
    return np.vstack((y[1], alpha_val * y[0] - (GSef_val / EIa_val) * p_expr(x)))

def bc_pk(ya, yb):
    return np.array([ya[0], yb[0]])  # Boundary: pk(0)=0, pk(L)=0

y_init = np.zeros((2, x_vals.size))
sol_pk = solve_bvp(
    ode_pk,
    bc_pk,
    x_vals,
    y_init,
    tol=1e-3,          # decrease tolerance for higher precision
    max_nodes=10000000   # allow more subdivision points
)
if not sol_pk.success:
    print("WARNING: BVP solver did not converge with default settings.")
    print(f"Message: {sol_pk.message}")
    print("→ Try increasing 'max_nodes' or relaxing 'tol' if convergence remains difficult.")
    raise RuntimeError("Failed to solve p_k(x)")

p_k_vals = sol_pk.sol(x_vals)[0]
p_a_vals = p_expr(x_vals) - p_k_vals    # load on beam A
p_b_vals = p_k_vals                     # load on beam B

print("Step 1: p_k(x), p_a(x), p_b(x) --- OK")


# ===============================================================
# STEP 2 — SOLVE FOR BEAM A: w_a'''' = p_a / EIa
# Boundary conditions: w(0)=0, w(L)=0, w''(0)=0, w''(L)=0
# ===============================================================

p_a_interp = lambda xx: np.interp(xx, x_vals, p_a_vals)

def ode_wa(x, y):
    # y = [w, w', w'', w''']
    return np.vstack((y[1], y[2], y[3], -p_a_interp(x) / EIa_val))

def bc_wa(ya, yb):
    return np.array([ya[0], yb[0], ya[2], yb[2]])

y_init = np.zeros((4, x_vals.size))
sol_wa = solve_bvp(
    ode_wa,
    bc_wa,
    x_vals,
    y_init,
    tol=1e-3,  # decrease tolerance for higher precision
    max_nodes=100000  # allow more subdivision points
)
if not sol_wa.success:
    print("WARNING: BVP solver did not converge with default settings.")
    print(f"Message: {sol_wa.message}")
    print("→ Try increasing 'max_nodes' or relaxing 'tol' if convergence remains difficult.")
    raise RuntimeError("Failed to solve w_a(x)")


w_a_vals = sol_wa.sol(x_vals)[0]
M_a_vals = EIa_val * sol_wa.sol(x_vals)[2]          # Bending moment M = EI * w''
T_a_vals = KR_dMdx(M_a_vals, x_vals)    # Shear force T = dM/dx

print("Step 2: w_a(x), M_a(x), T_a(x) --- OK")


# ===============================================================
# STEP 3 — SOLVE FOR BEAM B
# w_b,M'''' = p_b/EIb
# w_b,T'''' = (1/GSef) * p_b''
# ===============================================================

# Second derivative of p_b(x)
p_b_dd = np.gradient(np.gradient(p_b_vals, x_vals), x_vals)

# Interpolation functions
p_b_interp = lambda xx: np.interp(xx, x_vals, p_b_vals)
p_b_dd_interp = lambda xx: np.interp(xx, x_vals, p_b_dd)

# ---- Bending part: w_b,M ----
def ode_wbM(x, y):
    return np.vstack((y[1], y[2], y[3], -p_b_interp(x) / EIb_val))

def bc_wbM(ya, yb):
    return np.array([ya[0], yb[0], ya[2], yb[2]])

y_init = np.zeros((4, x_vals.size))
sol_wbM = solve_bvp(
    ode_wbM,
    bc_wbM,
    x_vals,
    y_init,
    tol = 1e-3,  # decrease tolerance for higher precision
    max_nodes = 100000  # allow more subdivision points
)
if not sol_wbM.success:
    print("WARNING: BVP solver did not converge with default settings.")
    print(f"Message: {sol_wbM.message}")
    print("→ Try increasing 'max_nodes' or relaxing 'tol' if convergence remains difficult.")
    raise RuntimeError("Failed to solve w_b,M(x)")

# ---- Shear part: w_b,T ----
def ode_wbT(x, y):
    # w_b,T'''' = +p_b''/GSef, NOT -p_b''/GSef.
    # The bending branch uses w'''' = -p/EI, which sets the sign convention
    # (w < 0 downwards, M = EI*w'' > 0 in sagging). The shear branch only
    # involves two integrations instead of four, so the same leading minus
    # would make the shear deflection come out UPWARDS while the bending one
    # goes down. With the minus in place, w_b,T was +0.757 mm where w_b,M was
    # -0.076 mm, and their honest sum (+0.681) bore no relation to
    # w_a = -0.833; the abs() that used to be applied further down hid this by
    # forcing both contributions to add downwards. With the sign below, the
    # two branches agree on their own: w_a - (w_b,M + w_b,T) = 2.5e-6.
    return np.vstack((y[1], y[2], y[3], p_b_dd_interp(x) / GSef_val))

def bc_wbT(ya, yb):
    return np.array([ya[0], yb[0], ya[2], yb[2]])

y_init = np.zeros((4, x_vals.size))
sol_wbT = solve_bvp(
    ode_wbT,
    bc_wbT,
    x_vals,
    y_init,
    tol = 1e-3,  # decrease tolerance for higher precision
    max_nodes = 100000  # allow more subdivision points
)

if not sol_wbT.success:
    print("WARNING: BVP solver did not converge with default settings.")
    print(f"Message: {sol_wbT.message}")
    print("→ Try increasing 'max_nodes' or relaxing 'tol' if convergence remains difficult.")
    raise RuntimeError("Failed to solve w_b,T(x)")

# Superposition of deflections.
# Plain sum, no abs(): taking absolute values forced the bending and shear
# contributions to add downwards whatever their actual signs. That happens to
# give the right answer for a load that pushes the same way over the whole
# span, but it is wrong as soon as the load changes sign along the beam - for
# p(x) = sin(2*pi*x/L), abs() dragged the upward half downwards and the
# w_a vs w_b check jumped from 1.3e-6 to 2.1e-1.
w_bM_vals = sol_wbM.sol(x_vals)[0]
w_bT_vals = sol_wbT.sol(x_vals)[0]
w_b_vals = w_bM_vals + w_bT_vals

M_b_vals = EIb_val * sol_wbM.sol(x_vals)[2]
T_b_vals = KR_dMdx(M_b_vals, x_vals)

print("Step 3: w_b,M(x), w_b,T(x), M_b(x), T_b(x) --- OK")


# ===============================================================
# STEP 4 — CONSISTENCY CHECK BETWEEN w_a AND w_b
# w_a = w_b
# ===============================================================
# Compute the maximum absolute difference and compare to tolerance
# Relative tolerance: an absolute 1e-3 mm threshold is meaningless on its own,
# since it is 0.1 % of a 1 mm deflection but only 0.002 % of a 50 mm one. The
# check is scaled by the largest deflection instead.
tolerance = 1e-3  # user-defined RELATIVE tolerance (can be adjusted)
w_scale = max(np.max(np.abs(w_a_vals)), 1e-12)
max_diff = np.max(np.abs(w_a_vals - w_b_vals)) / w_scale

if max_diff > tolerance:
    print(f"WARNING: Deflection mismatch detected. |w_a - w_b|_max / |w|_max = {max_diff:.3e} > {tolerance:.3e}")
    print("→ Consider increasing the number of discretization points or reducing solver tolerance.")
    print("Step 4: Consistency check --- KO")
else:
    print(f"Check OK: deflections consistent within relative tolerance ({max_diff:.3e} <= {tolerance:.3e})")
    print("Step 4: Consistency check --- OK")


# ===============================================================
# STEP 5 — COMPUTE & EXPORT EFFORTS, LOADS AND DEFORMATIONS
# ===============================================================
out_folder = "RES_all"
os.makedirs(out_folder, exist_ok=True)

p_tot_vals = p_a_vals + p_b_vals
M_tot_vals = M_a_vals + M_b_vals
T_tot_vals = T_a_vals + T_b_vals
w_vals = -w_a_vals

df = pd.DataFrame({
    "x [mm]": x_vals,
    "w_KR [mm]": w_vals,
    "p_a [N/mm]": p_a_vals,
    "p_b [N/mm]": p_b_vals,
    "p_tot [N/mm]": p_tot_vals,
    "M_a [Nmm]": M_a_vals,
    "M_b [Nmm]": M_b_vals,
    "M_tot [Nmm]": M_tot_vals,
    "T_a [N]": T_a_vals,
    "T_b [N]": T_b_vals,
    "T_tot [N]": T_tot_vals
})

filename = os.path.join(out_folder, f"all_{name}.csv")
df.to_csv(filename, sep="\t", index=False)

i_wmax = int(np.argmax(np.abs(w_vals)))
print(f' -> Maximal deflection : wmax = {w_vals[i_wmax]} [mm] at x = {x_vals[i_wmax]} [mm]')
print("Step 5: Compute and Export --- OK")

# ===============================================================
# STEP 6 — STRESS COMPUTATION FOR BEAM A
# ===============================================================
M_A = float(np.interp(x0_sigma, x_vals, M_a_vals))
A_Sigma, A_Sigma_zglobal = KR_A_Sigma(M_A, h, E, z, EIa_val)

T_A = float(np.interp(x0_tau, x_vals, T_a_vals))
A_Tau, A_Tau_zglobal = KR_A_Tau(T_A, h, E, z, EIa_val)

print("Step 6: Beam A stresses --- OK")


# ===============================================================
# STEP 7 — STRESS COMPUTATION FOR BEAM B
# ===============================================================

M_B = float(np.interp(x0_sigma, x_vals, M_b_vals))
B_Sigma, B_Sigma_zglobal = KR_B_Sigma(M_B, h, E, z, EIb_val)

T_B = float(np.interp(x0_tau, x_vals, T_b_vals))
B_Tau, B_Tau_zglobal = KR_B_Tau(T_B, h, E, z, EIb_val)

print("Step 7: Beam B stresses --- OK")


# ===============================================================
# STEP 8 — TOTAL STRESSES (SUM OF A AND B)
# ===============================================================
out_folder = "RES_stresses"
os.makedirs(out_folder, exist_ok=True)

# NORMAL STRESSES
# filename = 'RES_stresses\_stresses_' + name + '_SIGMA.csv'
# COMB_Sigma = sum_data(A_Sigma, B_Sigma)
# COMB_Sigma.to_csv(filename, sep='\t', index=False)

filename = os.path.join(out_folder, f"stresses_{name}_SIGMA_zglobal.csv")
COMB_Sigma_zglobal = sum_data(A_Sigma_zglobal, B_Sigma_zglobal)
COMB_Sigma_zglobal.to_csv(filename, sep='\t', index=False)


A_Sigma_zglobal_all = []
B_Sigma_zglobal_all = []
for val in x_vals:
    M_Ai = float(np.interp(val, x_vals, M_a_vals))
    A_Sigma_i, A_Sigma_zglobal_i = KR_A_Sigma(M_Ai, h, E, z, EIa_val)
    A_Sigma_zglobal_all.append(A_Sigma_zglobal_i)

    M_Bi = float(np.interp(val, x_vals, M_b_vals))
    B_Sigma_i, B_Sigma_zglobal_i = KR_B_Sigma(M_Bi, h, E, z, EIb_val)
    B_Sigma_zglobal_all.append(B_Sigma_zglobal_i)

SIGMA_MAX_L = KR_sigma_max_per_lamella(x_vals,A_Sigma_zglobal_all, B_Sigma_zglobal_all, h, z)
filename = os.path.join(out_folder, f"stresses_{name}_SIGMA_MAX_L.csv")
SIGMA_MAX_L.to_csv(filename, sep='\t', index=False)

# SHEAR STRESSES
# filename = 'RES_stresses\_stresses_' + name + '_TAU.csv'
# COMB_Tau = sum_data(A_Tau, B_Tau)
# COMB_Tau.to_csv(filename, sep='\t', index=False)

filename = os.path.join(out_folder, f"stresses_{name}_TAU_zglobal.csv")
COMB_Tau_zglobal = sum_data(A_Tau_zglobal, B_Tau_zglobal)
COMB_Tau_zglobal.to_csv(filename, sep='\t', index=False)


A_tau_zglobal_all = []
B_tau_zglobal_all = []
for val in x_vals:
    T_Ai = float(np.interp(val, x_vals, T_a_vals))
    A_tau_i, A_tau_zglobal_i = KR_A_Tau(T_Ai, h, E, z, EIa_val)
    A_tau_zglobal_all.append(A_tau_zglobal_i)

    T_Bi = float(np.interp(val, x_vals, T_b_vals))
    B_tau_i, B_tau_zglobal_i = KR_B_Tau(T_Bi, h, E, z, EIb_val)
    B_tau_zglobal_all.append(B_tau_zglobal_i)

tau_MAX_L = KR_tau_max_per_lamella(x_vals,A_tau_zglobal_all, B_tau_zglobal_all, h, z)
filename = os.path.join(out_folder, f"stresses_{name}_TAU_MAX_L.csv")
tau_MAX_L.to_csv(filename, sep='\t', index=False)

print("Step 8: Combined stresses --- OK")


# ===============================================================
# STEP 9 — SHEAR FORCES IN CONNECTORS
# ===============================================================
out_folder = "RES_interface"
os.makedirs(out_folder, exist_ok=True)

filename = os.path.join(out_folder, f"Ts_{name}.csv")
Ts_vals, df_Ts = KR_Ts(x_vals, T_b_vals, h, E, z, b, EIb_val)
df_Ts.to_csv(filename, sep='\t', index=False)

df_Fcon, Fconn_max = KR_ConnectorForces(x_vals, df_Ts, L_beam, s)
filename = os.path.join(out_folder, f"Fconn_{name}.csv")
df_Fcon.to_csv(filename, sep='\t', index=False)

for _j, _col in enumerate([c for c in df_Ts.columns if c != 'x [mm]']):
    print(f" -> Maximum connector load - {_col.replace('Ts_', 'Fconn').split(' [')[0]} :"
          f" {Fconn_max[_j]} [N]")

print("Step 9: Shear forces in connectors --- OK")


# ===============================================================
# STEP 10 — CONSISTENCY CHECK OF RESULTS
# ===============================================================
T0 = T_tot_vals[0]
TL = T_tot_vals[-1]
T = (T0 - TL)
Integraleq = np.trapezoid(p_tot_vals, x_vals)

ReactionF = Integraleq
err = abs(ReactionF - T) / abs(T) if abs(T) > 0 else np.inf
if err < 1e-3:
    print(f"Step 10: Consistency Check - err = {err:.3e} < e-3 --- OK")
else :
    print(f"Step 10: Consistency Check - err = {err:.3e} > e-3 --- KO")



print(f'============================================ end ==============================================')


# ===============================================================
# VISUALIZATION
# ===============================================================

plt.figure(figsize=(10, 6))
plt.plot(x_vals, p_tot_vals, 'k:',label='p_tot [N/mm]')
plt.plot(x_vals, p_a_vals, label='p_a [N/mm]')
plt.plot(x_vals, p_b_vals, label='p_b [N/mm]')
plt.xlabel("x [mm]")
plt.ylabel("Load [N/mm]")
plt.legend()
plt.title("Distributed loads on beams A and B")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_vals, w_a_vals,  label='w_a [mm]')
plt.plot(x_vals, w_b_vals,  label='w_b [mm]')
plt.plot(x_vals, w_bM_vals, label='w_bM [mm]')
plt.plot(x_vals, w_bT_vals, label='w_bT [mm]')
plt.xlabel("x [mm]")
plt.ylabel("Deflection [mm]")
plt.legend()
plt.title("Deflection of beams A and B")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_vals, M_tot_vals, 'k:', label='M_tot [Nmm]')
plt.plot(x_vals, M_a_vals, label='M_a [Nmm]')
plt.plot(x_vals, M_b_vals, label='M_b [Nmm]')
plt.xlabel("x [mm]")
plt.ylabel("Moment [Nmm]")
plt.legend()
plt.grid(True)
plt.title("Bending moments")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_vals, T_tot_vals, 'k:', label='T_tot [N]')
plt.plot(x_vals, T_a_vals, label='T_a [N]')
plt.plot(x_vals, T_b_vals, label='T_b [N]')
plt.xlabel("x [mm]")
plt.ylabel("Shear force [N]")
plt.legend()
plt.grid(True)
plt.title("Shear forces")
plt.show()
