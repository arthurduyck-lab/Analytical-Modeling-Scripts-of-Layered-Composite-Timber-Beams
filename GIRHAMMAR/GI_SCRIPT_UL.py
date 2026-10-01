# ===============================================================
# FUNCTIONS & IMPORTS
# ===============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from GI_FUNCTIONS import (read_data, GI_Parameters, GI_ConnectorForces,
                          GI_SIGMA, GI_SIGMA_global, GI_TAU, GI_TAU_global,
                          _trapezoid)



# ===============================================================
# INPUT DATA
# ===============================================================

# NAME CONFIG
name = 'GI_UL'


## LAMELLAE PARAMETERS
L_beam = 1400.0         # Beam length [mm]

## LOAD PARAMETERS
qm = 1.0                # [N/mm]

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

# Load
F0 = 0                   # Initial axial load[N]

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
def q_expr(x):
    """Uniform load"""
    return np.full_like(np.atleast_1d(x), qm, dtype=float)


# ===============================================================
# STEP 0 — COMPUTE BEAM PARAMETERS AND RIGIDITIES
# ===============================================================
(h, b, E, G, EA, EI, n, r, r_tot,
 EA0, EAp, z0, zCG, zcg_full, EI0, EI_full, K, s,
 alpha, alpha_carre, beta, gamma) = GI_Parameters(LAM, CON)

print("Step 0: alpha, beta, gamma --- OK")


# ===============================================================
# STEP 1 — SOLVE: w(x) (numerically)
# w'''''' - alpha**2 * w'''' = q''/EI0 - alpha**2 * q/EI_full
#
# Boundary conditions :
# w(0) = 0
# w''(0) = 0
# w''''(0) = q/EI_zero
# w(L) = 0
# w''(L) = 0
# w''''(L) = q/EI_zero
# ===============================================================

q_vals = q_expr(x_vals)

# Derivative of p(x)
q_d_vals = np.gradient(q_vals, x_vals)
q_dd_vals = np.gradient(q_d_vals, x_vals)

# Interpolation functions
q_interp = lambda xx: np.interp(xx, x_vals, q_vals)
q_dd_interp = lambda xx: np.interp(xx, x_vals, q_dd_vals)

# ----------------------------------------------------------------
# Convert sixth-order ODE to first-order system (6×1)
# y0=w, y1=w', y2=w'', y3=w''', y4=w'''', y5=w'''''
# y0' = y1
# y1' = y2
# y2' = y3
# y3' = y4
# y4' = y5
# y5' = alpha**2*y4 + (p_dd/EI0) - (alpha**2/EI_full)*p
# ----------------------------------------------------------------
def ode_w(x, y):
    qv = q_interp(x)
    qdd = q_dd_interp(x)
    dydx = np.zeros_like(y)
    dydx[0] = y[1]
    dydx[1] = y[2]
    dydx[2] = y[3]
    dydx[3] = y[4]
    dydx[4] = y[5]
    dydx[5] = alpha**2 * y[4] + (qdd / EI0) - (alpha**2 / EI_full) * qv
    return dydx

# Boundary conditions
def bc_w(ya, yb):
    return np.array([
        ya[0],                # w(0)=0
        ya[2],                # w''(0)=0
        ya[4] - q_vals[0] / EI0,   # w''''(0)=q(0)/EI0
        yb[0],                # w(L)=0
        yb[2],                # w''(L)=0
        yb[4] - q_vals[-1] / EI0   # w''''(L)=q(L)/EI0
    ])

# Initial guess
y_init = np.zeros((6, x_vals.size))

# Solve BVP
sol_w = solve_bvp(
    ode_w,
    bc_w,
    x_vals,
    y_init,
    tol=1e-3,
    max_nodes=1000000
)
if not sol_w.success:
    print("WARNING: BVP solver did not converge with default settings.")
    print(f"Message: {sol_w.message}")
    print("→ Try increasing 'max_nodes' or relaxing 'tol' if convergence remains difficult.")
    raise RuntimeError("Failed to solve w(x)")

w_vals = sol_w.sol(x_vals)[0]
w_d_vals = sol_w.sol(x_vals)[1]
w_dd_vals = sol_w.sol(x_vals)[2]
w_ddd_vals = sol_w.sol(x_vals)[3]
w_dddd_vals = sol_w.sol(x_vals)[4]
w_ddddd_vals = sol_w.sol(x_vals)[5]


# ===============================================================
# STEP 2 - COMPUTE & EXPORT EFFORTS, LOADS AND DEFORMATIONS
#
# w comes from solve_bvp, so w'''' and w''''' below are the TRUE derivatives.
# The exact Girhammar relations therefore apply as such, q-term included:
#     M = EI_full/alpha**2 * w''''  - EI_full * w''  - EI_full/(alpha**2*EI0) * q
#     T = EI_full/alpha**2 * w''''' - EI_full * w''' - EI_full/(alpha**2*EI0) * q'
#     Ts = 1/r_tot * (M' + EI0 * w''')
#
# Do NOT transpose these formulas to the modal scripts GI_SCRIPT_FS-3Pts.py and
# GI_SCRIPT_FS-4Pts.py: there w'''' is piecewise-smoothed and the q-term has
# already been cancelled analytically. See the header of their STEP 3.
#
#     N1 = 1/r_tot * [F0 * (r_tot - zcg_full) + M + EI0 * w'']
#     M1 = -EI[0] * w''
#     T1 = M1' + Ts * r1
#
#     N2 = 1/r_tot * [F0 * zcg_full - M - EI0 * w'']
#     M2 = -EI[1] * w''
#     T2 = M2' + Ts * r2
# ===============================================================
out_folder = "RES_all"
os.makedirs(out_folder, exist_ok=True)

M_vals = EI_full / alpha_carre * w_dddd_vals - EI_full * w_dd_vals - EI_full / (alpha_carre*EI0) * q_vals
M_d_vals = np.gradient(M_vals, x_vals)

T_vals = EI_full / alpha_carre * w_ddddd_vals - EI_full * w_ddd_vals - EI_full / (alpha_carre*EI0) * q_d_vals

Ts_vals = 1/r_tot * (M_d_vals + EI0 * w_ddd_vals)

M1_vals = -EI[0] * w_dd_vals
N1_vals = 1/r_tot * (F0 * (r_tot - zcg_full) + M_vals + EI0 * w_dd_vals)
T1_vals = -EI[0] * w_ddd_vals + Ts_vals * r[0]

M2_vals = -EI[1] * w_dd_vals
N2_vals = 1/r_tot * (F0 * zcg_full - M_vals - EI0 * w_dd_vals)
T2_vals = -EI[1] * w_ddd_vals + Ts_vals * r[1]

df = pd.DataFrame({
    "x [mm]": x_vals,
    "q [N/mm]": q_vals,
    "w [mm]": w_vals,
    "M [Nmm]": M_vals,
    "T [N]": T_vals,
    "Ts [N/mm]": Ts_vals,
    "M1 [Nmm]": M1_vals,
    "N1 [N]": N1_vals,
    "T1 [N]": T1_vals,
    "M2 [Nmm]": M2_vals,
    "N2 [N]": N2_vals,
    "T2 [N]": T2_vals
})


filename = os.path.join(out_folder, f"all_{name}.csv")
df.to_csv(filename, sep="\t", index=False)

i_wmax = int(np.argmax(np.abs(w_vals)))
print(f' -> Maximal deflection : wmax = {w_vals[i_wmax]} [mm] at x = {x_vals[i_wmax]} [mm]')
print("Step 2: Compute and Export --- OK")


# ===============================================================
# STEP 3 — STRESS COMPUTATION FOR BEAM
# ===============================================================
out_folder = "RES_stresses"
os.makedirs(out_folder, exist_ok=True)

# Normal
M_1 = float(np.interp(x0_sigma, x_vals, M1_vals))
N_1 = float(np.interp(x0_sigma, x_vals, N1_vals))

M_2 = float(np.interp(x0_sigma, x_vals, M2_vals))
N_2 = float(np.interp(x0_sigma, x_vals, N2_vals))

MN1_sigma_zlocal = GI_SIGMA(1, N_1, M_1, h, b)
MN2_sigma_zlocal = GI_SIGMA(2, N_2, M_2, h, b)

SigmaTOT_zglobal = GI_SIGMA_global(z0, zCG, MN1_sigma_zlocal, MN2_sigma_zlocal)
filename = os.path.join(out_folder, f"stresses_{name}_SIGMA_zglobal.csv")
SigmaTOT_zglobal.to_csv(filename, sep="\t", index=False)

# Sigma max along the beam
sigmaMAX_L1 = []
sigmaMAX_L2 = []
for val in x_vals :
    M1_i = float(np.interp(val, x_vals, M1_vals))
    N1_i = float(np.interp(val, x_vals, N1_vals))
    M2_i = float(np.interp(val, x_vals, M2_vals))
    N2_i = float(np.interp(val, x_vals, N2_vals))

    sigma1_i = GI_SIGMA(1, N1_i, M1_i, h, b)
    sigma_vals1 = sigma1_i['σTOT [MPa]']
    idx1 = np.argmax(np.abs(sigma_vals1))
    sigmamax_signed1 = sigma_vals1[idx1]
    sigmaMAX_L1.append(sigmamax_signed1)

    sigma2_i = GI_SIGMA(2, N2_i, M2_i, h, b)
    sigma_vals2 = sigma2_i['σTOT [MPa]']
    idx2 = np.argmax(np.abs(sigma_vals2))
    sigmamax_signed2 = sigma_vals2[idx2]
    sigmaMAX_L2.append(sigmamax_signed2)

df_sigmaMAX_L = pd.DataFrame({
    "x [mm]": x_vals,
    "σmax_1 [MPa]": sigmaMAX_L1,
    "σmax_2 [MPa]": sigmaMAX_L2
})

filename = os.path.join(out_folder, f"stresses_{name}_SIGMA_MAX_L.csv")
df_sigmaMAX_L.to_csv(filename, sep="\t", index=False)


# Shear
T_12 = float(np.interp(x0_tau, x_vals, T_vals))
Ts_12 = float(np.interp(x0_tau, x_vals, Ts_vals))

tau1_zlocal = GI_TAU(1, T_12, Ts_12, h, b, E, r_tot, r, EI0)
tau2_zlocal = GI_TAU(2, T_12, Ts_12, h, b, E, r_tot, r, EI0)
TauTOT_zglobal = GI_TAU_global(z0, zCG, tau1_zlocal, tau2_zlocal)

filename = os.path.join(out_folder, f"stresses_{name}_TAU_zglobal.csv")
TauTOT_zglobal.to_csv(filename, sep="\t", index=False)

# Shear max along the beam
tauMAX_L1 = []
tauMAX_L2 = []
for val in x_vals :
    T12_i = float(np.interp(val, x_vals, T_vals))
    Ts12_i = float(np.interp(val, x_vals, Ts_vals))

    tau1_i = GI_TAU(1, T12_i, Ts12_i, h, b, E, r_tot, r, EI0)
    tau_vals1 = tau1_i['τ [MPa]']
    idx1 = np.argmax(np.abs(tau_vals1))
    taumax_signed1 = tau_vals1[idx1]
    tauMAX_L1.append(taumax_signed1)

    tau2_i = GI_TAU(2, T12_i, Ts12_i, h, b, E, r_tot, r, EI0)
    tau_vals2 = tau2_i['τ [MPa]']
    idx2 = np.argmax(np.abs(tau_vals2))
    taumax_signed2 = tau_vals2[idx2]
    tauMAX_L2.append(taumax_signed2)

df_tauMAX_L = pd.DataFrame({
    "x [mm]": x_vals,
    "τmax_1 [MPa]": tauMAX_L1,
    "τmax_2 [MPa]": tauMAX_L2
})
filename = os.path.join(out_folder, f"stresses_{name}_TAU_MAX_L.csv")
df_tauMAX_L.to_csv(filename, sep="\t", index=False)

print("Step 3: Beam stresses --- OK")

# ===============================================================
# STEP 4 — SHEAR FORCES IN CONNECTORS
# ===============================================================
out_folder = "RES_interface"
os.makedirs(out_folder, exist_ok=True)

filename = os.path.join(out_folder, f"Ts_{name}.csv")
df_Ts = pd.DataFrame({
    "x [mm]": x_vals,
    "Ts [N/mm]": Ts_vals,
})
df_Ts.to_csv(filename, sep="\t", index=False)

df_Fcon = GI_ConnectorForces(x_vals, df_Ts, L_beam, s)
filename = os.path.join(out_folder, f"Fconn_{name}.csv")
df_Fcon.to_csv(filename, sep='\t', index=False)

print("Step 4: Shear Forces in connectors --- OK")


# ===============================================================
# STEP 5 — CONSISTENCY CHECK OF RESULTS
# ===============================================================
T0 = T_vals[0]
TL = T_vals[-1]
T = (T0 - TL)

Integraleq = _trapezoid(q_vals, x_vals)

ReactionF = Integraleq
err = abs(ReactionF - T) / abs(T) if abs(T) > 0 else np.inf
if err < 1e-3:
    print(f"Step 5: Consistency Check - err = {err:.3e} < e-3 --- OK")
else :
    print(f"Step 5: Consistency Check - err = {err:.3e} > e-3 --- KO")

print(f'============================================ end ==============================================')

# ===============================================================
# VISUALIZATION
# ===============================================================
plt.figure(figsize=(8,4))
plt.plot(x_vals, q_vals, label='p (N/mm)')
plt.xlabel('x (mm)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(x_vals, w_vals, label='w (mm)')
plt.xlabel('x (mm)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(x_vals, M_vals, label='M (Nmm)')
plt.xlabel('x (mm)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(x_vals, T_vals, label='T (N)')
plt.xlabel('x (mm)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(x_vals, Ts_vals, label='Ts (N/mm)')
plt.xlabel('x (mm)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



## 1
# Bending moment M1
fig, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(x_vals, M1_vals, color='tab:blue', label='M1 (Nmm)')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('M1 (Nmm)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Normal & shear forces
ax2 = ax1.twinx()
ax2.plot(x_vals, N1_vals, color='tab:orange', label='N1 (N)')
ax2.plot(x_vals, T1_vals, color='tab:green', label='T1 (N)')
ax2.set_ylabel('N1, T1 (N)', color='tab:gray')
ax2.tick_params(axis='y', labelcolor='tab:gray')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.show()


## 2
# Bending moment M2
fig, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(x_vals, M2_vals, color='tab:blue', label='M2 (Nmm)')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('M2 (Nmm)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Normal & shear forces
ax2 = ax1.twinx()
ax2.plot(x_vals, N2_vals, color='tab:orange', label='N2 (N)')
ax2.plot(x_vals, T2_vals, color='tab:green', label='T2 (N)')
ax2.set_ylabel('N2, T2 (N)', color='tab:gray')
ax2.tick_params(axis='y', labelcolor='tab:gray')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.show()