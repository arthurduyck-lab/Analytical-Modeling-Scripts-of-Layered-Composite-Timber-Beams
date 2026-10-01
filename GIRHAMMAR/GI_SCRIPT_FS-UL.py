# ===============================================================
# FUNCTIONS & IMPORTS
# ===============================================================

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from GI_FUNCTIONS import (read_data, GI_Parameters, GI_ConnectorForces,
                          GI_SIGMA, GI_SIGMA_global, GI_TAU, GI_TAU_global,
                          GI_MODAL_from_FS, GI_piecewise_derivatives, _trapezoid)


# ===============================================================
# INPUT DATA
# ===============================================================
## NAME CONFIG
name = 'GI_FS-UL'

## LAMELLAE PARAMETERS
L_beam = 1400.0         # Beam length [mm]

## LOAD PARAMETERS
qm = 1.0                 # [N]


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
F0 = 0                  # Initial axial load[N]
F_position = L_beam/2
l = L_beam                  # Length of load application [mm]
d = F_position - l/2    # Length btw support and load [mm]
e = F_position + l/2


## PRINT DATA
print(f'============================================ {name} ==============================================')
print(f'Force : qm = {qm} [N/mm]')
print(f'Length of the beam : L = {L_beam} [mm]')
print(f'Position of the force : F_pos = {F_position} [mm]')
print('\n')

print(f'Length of load application : l = {l} [mm]')
print(f'Linear load : qm = {qm} [N/mm]')
print('\n')

print(f'Lamellae properties')
print(LAM)
print('\n')

print(f'Connector properties properties')
print(CON)
print('\n')


## DATA CONSISTENCY CHECK
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

xsupp_vals = [d, e, F_position]
for val in xsupp_vals:
    if not (x_vals == val).any():
        x_vals = np.append(x_vals, val)

x_vals = np.sort(x_vals)


# ===============================================================
# STEP 0 — COMPUTE BEAM PARAMETERS AND RIGIDITIES
# ===============================================================
(h, b, E, G, EA, EI, n, r, r_tot,
 EA0, EAp, z0, zCG, zcg_full, EI0, EI_full, K, s,
 alpha, alpha_carre, beta, gamma) = GI_Parameters(LAM, CON)

print("Step 0: alpha, beta, gamma --- OK")

# ===============================================================
# STEP 1 — SOLVE: w(x) (modal approach)
# Modal decomposition based on orthogonal sine modes:
#
#     q(x) = 2/pi * q_m * Σ[C_t/t * sin(k_t * x)]
#       with k_t = t * pi/L    &    A_t = 2/pi * q_m * C_t/t
#     q(x) = Σ [ A_t * sin(k_t * x) ]
#
#     w(x) = Σ [ B_t * sin(k_t * x) ]
#
# Substitution into governing equation:
#     w'''''' - α² * w'''' = q''/EI₀ - α² * q/EI_full
#     EI₀ * Σ[B_t * k_t⁴ * sin(k_tx)] - EI₀/EI_full * α² * Σ[B_t * k_t² * sin(k_tx)] + Σ[A_t * sin(k_tx)]
#     B_t = A_t * S_t /(k_t⁴ * (k_t² + α²))
#       with S_t = k_t²/EI₀ + α²/EI_full
#
# Projection onto each mode using orthogonality on [0, L]:
#     w_t(x) = B_t * sin(k_t * x)
#     w(x) = Σ[B_t * sin(k_t * x)]
#
# Boundary conditions are automatically satisfied by sine modes:
#     w(0) = w(L) = 0
# ===============================================================
out_folder = "RES_all"
os.makedirs(out_folder, exist_ok=True)

N = 5000
t = np.arange(1, N+1)

Ct = np.cos(t * d * np.pi / L_beam) - np.cos(t * (d + l) * np.pi / L_beam)  # FS coefficients for 3Pts flexural test

res = GI_MODAL_from_FS(x_vals, qm, L_beam, Ct, N, alpha, EI0, EI_full)

w_vals = res['w']
dw = res['w1']
dw2 = res['w2']
dw3 = res['w3']
dw4 = res['w4']
dw5 = res['w5']

q_vals = res['q']
q_d1_vals = res['qp']
q_d2_vals = res['qpp']

filename = os.path.join(out_folder, f"qw_derivates_{name}.csv")
dfqw_derivates = pd.DataFrame({
    'x [mm]': x_vals,
    'q [N/mm]' : q_vals,
    'q1 [N/mm]' : q_d1_vals,
    'q2 [N/mm]' : q_d2_vals,
    'w [mm]': w_vals,
    'w1 [mm]' : dw,
    'w2 [mm]' : dw2,
    'w3 [mm]': dw3,
    'w4 [mm]': dw4,
    'w5 [mm]': dw5
})
dfqw_derivates.to_csv(filename, sep="\t", index=False)


# ===============================================================
# STEP 2 — Derivatives of piecewise w(x) and smoothing
# ===============================================================
out_folder = "RES_all"
os.makedirs(out_folder, exist_ok=True)

intervals = [
    (0, F_position, True, False),   # [0, F_position[
    (F_position, L_beam, True, True)   # [F_position, L_beam]
]

n_deriv = 5
polynomial_order = 14   # >=12 required: 5 successive derivatives of a
                        # degree-n fit lose n-5 orders of shape information.
                        # Only safe because the fit is done on a normalised
                        # domain (see GI_piecewise_derivatives).
margin = l

derivs = GI_piecewise_derivatives(
    x_vals,
    w_vals,
    intervals,
    n_derivatives=n_deriv,
    margin = margin,
    poly_order= polynomial_order)

w_d1_vals, w_d2_vals, w_d3_vals, w_d4_vals, w_d5_vals = derivs

filename = os.path.join(out_folder, f"qw_derivates-smoothed_{name}.csv")
dfqw_derivates_sm = pd.DataFrame({
    'x [mm]': x_vals,
    'q [N/mm]' : q_vals,
    'q1 [N/mm]' : q_d1_vals,
    'q2 [N/mm]' : q_d2_vals,
    'w1 [mm]' : w_d1_vals,
    'w2 [mm]' : w_d2_vals,
    'w3 [mm]': w_d3_vals,
    'w4 [mm]': w_d4_vals,
    'w5 [mm]': w_d5_vals
})
dfqw_derivates_sm.to_csv(filename, sep="\t", index=False)

print("Step 2: Piecewise derivates --- OK")


# ===============================================================
# STEP 3 - COMPUTE & EXPORT EFFORTS, LOADS AND DEFORMATIONS
#
# WHY THE q-TERM IS PRESENT IN M HERE  -- read before modifying.
#
# The exact Girhammar relations are
#     M = EI_full/alpha**2 * w''''  - EI_full * w''  - EI_full/(alpha**2*EI0) * q
#     T = EI_full/alpha**2 * w''''' - EI_full * w''' - EI_full/(alpha**2*EI0) * q'
#
# Unlike GI_SCRIPT_FS-3Pts.py and GI_SCRIPT_FS-4Pts.py, the load is uniform:
# q/EI0 is a CONSTANT over the whole span, so the polynomial fit reproduces it
# and w''''_smoothed still contains it. The q-term must therefore be subtracted
# explicitly, exactly as in the solve_bvp scripts. Removing it would overestimate
# the mid-span moment by a factor of about 2.6.
#
# q_vals_cal is the exact constant qm rather than the truncated Fourier sum,
# to avoid Gibbs oscillations near the supports.
# q' = 0 for a uniform load, hence no q-term in T.
#
# Formulas actually used below:
#     M  = EI_full/alpha**2 * w''''  - EI_full * w'' - EI_full/(alpha**2*EI0) * qm
#     T  = EI_full/alpha**2 * w''''' - EI_full * w'''
#     Ts = 1/r_tot * (T + EI0 * w''')
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

q_vals_cal = np.full_like(q_vals, qm)
M_vals = EI_full / alpha_carre * w_d4_vals - EI_full * w_d2_vals - EI_full / (alpha_carre*EI0) * q_vals_cal
T_vals = EI_full / alpha_carre * w_d5_vals - EI_full * w_d3_vals


Ts_vals = 1/r_tot * (T_vals + EI0 * w_d3_vals)

M1_vals = -EI[0] * w_d2_vals
N1_vals = 1/r_tot * (F0 * (r_tot - zcg_full) + M_vals + EI0 * w_d2_vals)
T1_vals = -EI[0] * w_d3_vals + Ts_vals * r[0]

M2_vals = -EI[1] * w_d2_vals
N2_vals = 1/r_tot * (F0 * zcg_full - M_vals - EI0 * w_d2_vals)
T2_vals = -EI[1] * w_d3_vals + Ts_vals * r[1]

dfall = pd.DataFrame({
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
dfall.to_csv(filename, sep="\t", index=False)

i_wmax = int(np.argmax(np.abs(w_vals)))
print(f' -> Maximal deflection : wmax = {w_vals[i_wmax]} [mm] at x = {x_vals[i_wmax]} [mm]')
print("Step 3: Compute and Export --- OK")


# ===============================================================
# STEP 4 — STRESS COMPUTATION FOR BEAM
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

print("Step 4: Beam stresses --- OK")

# ===============================================================
# STEP 5 — SHEAR FORCES IN CONNECTORS
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

print("Step 5: Shear Forces in connectors --- OK")


# ===============================================================
# STEP 6 — CONSISTENCY CHECK OF RESULTS
# ===============================================================
T0 = T_vals[0]
TL = T_vals[-1]
T = (T0 - TL)

Integraleq = _trapezoid(q_vals, x_vals)   # kept for information only

# The Fourier reconstruction of a uniform load overshoots at the supports
# (Gibbs phenomenon), so integrating q_vals under-estimates the total load by
# about 1e-3 (1398.68 instead of 1400.00 for qm = 1, L = 1400). Using it as the
# reference makes this check fail on the reference value itself, not on T.
# The exact resultant is used instead, consistently with FS-3Pts (F) and
# FS-4Pts (2*P).
ReactionF = qm * L_beam
err = abs(ReactionF - T) / abs(T) if abs(T) > 0 else np.inf
if err < 1e-3:
    print(f"Step 6: Consistency Check - err = {err:.3e} < e-3 --- OK")
else :
    print(f"Step 6: Consistency Check - err = {err:.3e} > e-3 --- KO")

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