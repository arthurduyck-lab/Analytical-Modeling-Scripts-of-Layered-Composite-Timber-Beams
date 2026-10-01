# ===============================================================
# FUNCTIONS & IMPORTS
# ===============================================================

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from GI_gen_FUNCTIONS import (read_data, GI_gen_Parameters, GI_gen_MODAL,
                              GI_gen_SigmaSection, GI_gen_SigmaMaxi,
                              GI_gen_TauSection, GI_gen_TauMaxi, GI_gen_Ts,
                              GI_gen_ConnectorForces)


# ===============================================================
# INPUT DATA
# ===============================================================
## NAME CONFIG
name = 'GI-gen_FS-3Pts'

## LAMELLAE AND CONNECTORS
L_beam = 10000.0         # Beam length [mm]

## LOAD PARAMETERS
F = 2000       # [N]


## SECTION FOR STRESSES
x0_sigma = 1200.0         # Location for bending stress
x0_tau = 150.0            # Location for shear stress


## INTERMEDIATES PARAMETERS
# Lamellae
omega = np.pi / float(L_beam)

path_LAM = 'data_LAM.csv'      # tab-separated text file
path_CON = 'data_CON.csv'      # tab-separated text file
LAM = read_data(path_LAM, sep='\t')
CON = read_data(path_CON, sep='\t')

# Load
F_position = L_beam/2
l = 10                  # Length of load application [mm]
d = F_position - l/2    # Length btw support and load [mm]
e = F_position + l/2
qm = F/l                # [N/mm]


## PRINT DATA
print(f'============================================ {name} ==============================================')
print(f'Force : F = {F} [N]')
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
out = GI_gen_Parameters(LAM, CON, L_beam)

h = out['h']
b = out['b']
E = out['E']
G = out['G']
A = out['A']
I = out['I']
EA = out['EA']
EI = out['EI']
z0 = out['z0']
n = out['n']
c = out['c']
s = out['s']
K = out['K']
Kim1_i = out['Kim1_i']
Ki_ip1 = out['Ki_ip1']
v = out['v']
vim1_i = out['vim1_i']
vi_ip1 = out['vi_ip1']
print("Step 0: Beam Parameters --- OK")

# ===============================================================
# STEP 1 — SOLVE: w(x) and ui(x) (modal approach)
# a) ∑(i=1 ,n) EAi * ui' = 0
# b) -EAi * ui'' - (uip1 - ui + w' * vi_ip1) + (ui - uim1 + w' * vim1_i) = 0 		∀i ≠ s      with s = ⌊n/2 + 1⌋
# c) q = -w'''' *  ∑(i=1 ,n) EIi + ∑(i=1 ,n) (vi * (Ki_ip1 * (uip1' - ui' + w'' * vi_ip1) + Kim1 * (ui' - uim1' + w'' * vim1_i)))
#
# q(x) = A_t * sin(omega *x)
#           with A_t = 2/pi * q_m * C_t/t
#                C_t = Fourier Coefficients
# w(x) = w0 * sin(omega * x)
# ui(x) = u0i * cos(omega * x)
#           with omega = pi/L
# ===============================================================
out_folder = "RES_all"
os.makedirs(out_folder, exist_ok=True)

# Reference definition
lam_ref = math.floor(n/2) + 1
idx_lamref = lam_ref -1
idx_sup = []
idx_inf = []
idx_cent = []
for i in range(n):
    if i > idx_lamref :
        idx_sup.append(i)
    elif i < idx_lamref :
        idx_inf.append(i)
    elif i == idx_lamref:
        idx_cent.append(i)

idx_sup = np.array(idx_sup, dtype=int)
idx_inf = np.array(idx_inf, dtype=int)
idx_cent = np.array(idx_cent, dtype=int)
idx_s = idx_lamref

N_modes = 5000    # convergence reached around 500 for every mode tested
t_vals = np.arange(1, N_modes+1)
Ct_list = np.cos(t_vals * d * np.pi / L_beam) - np.cos(t_vals * (d + l) * np.pi / L_beam)  # FS coefficients for 3Pts flexural test

# Modes are accumulated in place. Storing every mode separately would allocate
# N_modes * (5 + 3n) * len(x_vals) floats (~560 MB for N_modes=5000, n=3, and
# over 2 GB for n=7 on a finer mesh) only to sum them immediately afterwards.
q_vals = np.zeros(len(x_vals))

w_vals = np.zeros(len(x_vals))
w_d1_vals = np.zeros(len(x_vals))
w_d2_vals = np.zeros(len(x_vals))
w_d3_vals = np.zeros(len(x_vals))

ui_vals = np.zeros((n, len(x_vals)))
ui_d1_vals = np.zeros((n, len(x_vals)))
ui_d2_vals = np.zeros((n, len(x_vals)))

# Loop on sinusoidal modes
for t in t_vals:
    k_t = t * np.pi / L_beam

    C_t = Ct_list[t - 1]
    A_t = (2 / np.pi) * qm * C_t / t

    q_vals += A_t * np.sin(k_t * x_vals)

    Asys_t, Bsys_t, ui0_t, w0_t = GI_gen_MODAL(
        n=n,
        idx_s=idx_s,
        EA=EA,
        EI=EI,
        q0=A_t,
        omega=k_t,
        v=v,
        vi_ip1=vi_ip1,
        vim1_i=vim1_i,
        Ki_ip1=Ki_ip1,
        Kim1_i=Kim1_i
    )

    w_vals += - w0_t * np.sin(k_t * x_vals)
    w_d1_vals += - w0_t * k_t * np.cos(k_t * x_vals)
    w_d2_vals += w0_t * k_t**2 * np.sin(k_t * x_vals)
    w_d3_vals += w0_t * k_t**3 * np.cos(k_t * x_vals)

    for i in range(n):
        ui_vals[i] += ui0_t[i] * np.cos(k_t * x_vals)
        ui_d1_vals[i] += - ui0_t[i] * k_t * np.sin(k_t * x_vals)
        ui_d2_vals[i] += - ui0_t[i] * k_t**2 * np.cos(k_t * x_vals)


# Export
data = {
    'x [mm]': x_vals,
    'q [N/mm]': q_vals,
    'w_GI [mm]': w_vals,
    'wd1 [mm]': w_d1_vals,
    'wd2 [mm]': w_d2_vals,
    'wd3 [mm]': w_d3_vals,
}
for i in range(n):
    data[f'ui{i+1} [mm]'] = ui_vals[i]
    data[f'ui{i + 1}_d1 [mm]'] = ui_d1_vals[i]
    data[f'ui{i + 1}_d2 [mm]'] = ui_d2_vals[i]


df_qwu = pd.DataFrame(data)
filename = os.path.join(out_folder, f"qwu_derivates_{name}.csv")
df_qwu.to_csv(filename, sep="\t", index=False)

print("Step 1: Solving w0 and u0 --- OK")



# ===============================================================
# STEP 2 — INTERNAL FORCES & DEFORMATIONS - COMPUTE & EXPORT
# ===============================================================
out_folder = "RES_all"
os.makedirs(out_folder, exist_ok=True)

def M_expr(x):
    x = np.atleast_1d(x)
    M = np.where(x < F_position, F/2 * x, F/2 * (L_beam - x))
    return np.full_like(x, M, dtype=float)
M_vals = M_expr(x_vals)

def T_expr(x):
    x = np.atleast_1d(x)
    T = np.where(x < F_position, F/2, -F/2)
    return np.full_like(x, T, dtype=float)
T_vals = T_expr(x_vals)

Mi_vals = [-w_d2_vals * EI[i] for i in range(n)]
Ni_vals = [ui_d1_vals[i] * EA[i] for i in range(n)]

dMi_vals = [-w_d3_vals * EI[i] for i in range(n)]
dNi_vals = [ui_d2_vals[i] * EA[i] for i in range(n)]

data_all ={
    'x [mm]': x_vals,
    'q [N/mm]': q_vals,
    'w_GI [mm]': w_vals,
    'M [Nmm]': M_vals,
    'T [N]': T_vals
}

for i in range(n):
    data_all[f'M{i + 1} [Nmm]'] = Mi_vals[i]

for i in range(n):
    data_all[f'N{i + 1} [N]'] = Ni_vals[i]


df_all = pd.DataFrame(data_all)
filename = os.path.join(out_folder, f"all_{name}.csv")
df_all.to_csv(filename, sep="\t", index=False)


i_wmax = int(np.argmax(np.abs(w_vals)))
print(f' -> Maximal deflection : wmax = {w_vals[i_wmax]} [mm] at x = {x_vals[i_wmax]} [mm]')
print("Step 2: Internal forces --- OK")

# ===============================================================
# STEP 3 — STRESS COMPUTATION FOR BEAM
# ===============================================================
out_folder = "RES_stresses"
os.makedirs(out_folder, exist_ok=True)

# Normal
M = [float(np.interp(x0_sigma, x_vals, Mi_vals[i])) for i in range(n)]
N = [float(np.interp(x0_sigma, x_vals, Ni_vals[i])) for i in range(n)]

df_sigmaSection = GI_gen_SigmaSection(M, N, h, b, z0)
filename = os.path.join(out_folder, f"stresses_{name}_SIGMA_zglobal.csv")
df_sigmaSection.to_csv(filename, sep="\t", index=False)

df_sigmaMAX_L = pd.DataFrame({'x [mm]': x_vals})
for i in range(n):
    sigmaTOT_max_list = []
    for val in x_vals:
        M_i = float(np.interp(val, x_vals, Mi_vals[i]))
        N_i = float(np.interp(val, x_vals, Ni_vals[i]))
        sigmaTOT_max_list.append(GI_gen_SigmaMaxi(i, M_i, N_i, h, b))
    df_sigmaMAX_L[f'σTOT_{i + 1} [MPa]'] = sigmaTOT_max_list

filename = os.path.join(out_folder, f"stresses_{name}_SIGMA_MAX_L.csv")
df_sigmaMAX_L.to_csv(filename, sep="\t", index=False)


# Shear
dM = [float(np.interp(x0_tau, x_vals, dMi_vals[i])) for i in range(n)]
dN = [float(np.interp(x0_tau, x_vals, dNi_vals[i])) for i in range(n)]

df_tauSection = GI_gen_TauSection(h, b, A, I, z0, dN, dM, idx_inf, idx_cent, idx_sup)
filename = os.path.join(out_folder, f"stresses_{name}_TAU_zglobal.csv")
df_tauSection.to_csv(filename, sep="\t", index=False)

tau_max_all = []
for val in x_vals:
    dM_i = [float(np.interp(val, x_vals, dMi_vals[i])) for i in range(n)]
    dN_i = [float(np.interp(val, x_vals, dNi_vals[i])) for i in range(n)]
    tau_max_i = GI_gen_TauMaxi(h, b, A, I, z0, dN_i, dM_i, idx_inf, idx_cent, idx_sup)
    tau_max_all.append(tau_max_i)

df_tauMAX_L = pd.DataFrame(tau_max_all, columns=[f"τTOT_{i+1} [MPa]" for i in range(n)])
df_tauMAX_L.insert(0, "x [mm]", x_vals)
filename = os.path.join(out_folder, f"stresses_{name}_TAU_MAX_L.csv")
df_tauMAX_L.to_csv(filename, sep="\t", index=False)

print("Step 3: Beam stresses --- OK")


# ===============================================================
# STEP 4 — SHEAR FORCES IN CONNECTORS
# ===============================================================
out_folder = "RES_interface"
os.makedirs(out_folder, exist_ok=True)

df_Ts = pd.DataFrame({'x [mm]': x_vals})

Ts_all = []
for val in x_vals:
    dM_i = [float(np.interp(val, x_vals, dMi_vals[i])) for i in range(n)]
    dN_i = [float(np.interp(val, x_vals, dNi_vals[i])) for i in range(n)]
    Ts_i = GI_gen_Ts(h, b, A, I, z0, dN_i, dM_i, idx_inf, idx_cent, idx_sup)
    Ts_all.append(Ts_i)

df_Ts = pd.DataFrame(Ts_all, columns=[f"Ts_{i+1},{i+2} [N/mm]" for i in range(n-1)])
df_Ts.insert(0, "x [mm]", x_vals)
filename = os.path.join(out_folder, f"Ts_{name}.csv")
df_Ts.to_csv(filename, sep="\t", index=False)


df_Fcon, Fconn_max = GI_gen_ConnectorForces(x_vals, df_Ts, L_beam, s)
filename = os.path.join(out_folder, f"Fconn_{name}.csv")
df_Fcon.to_csv(filename, sep='\t', index=False)

for i in range(n-1):
    print(f' -> Maximum connector load - Fconn{i+1},{i+2} : {Fconn_max[i]} [N]')

print("Step 4: Shear Forces in connectors --- OK")
# ===============================================================
# STEP 5 — CONSISTENCY CHECK OF RESULTS
# Sectional equilibrium of the layered cross-section:
#     M(x) = sum_i M_i(x) + sum_i N_i(x) * (zG - z_i)
# zG is the elastic centroid; since sum_i N_i = 0 the result does not
# actually depend on the reference level, but zG keeps the two terms of
# comparable magnitude. The sign of the axial contribution follows the
# convention sigma = -M*z/I used in GI_gen_SigmaSection.
#
# The check is evaluated away from the loaded zone: there M(x) is the
# idealised point-load diagram while the model is actually loaded by a
# patch of length l, and the two differ by qm*(l/2)**2/2 right under the
# load (1.9e-3 relative for the reference 3Pts case) — a modelling
# difference, not a numerical error.
# ===============================================================
zG_chk = np.sum(z0 * EA) / np.sum(EA)
sumN_chk = sum(Ni_vals[i] for i in range(n))
err_N = np.max(np.abs(sumN_chk)) / max(np.max(np.abs(Ni_vals[0])), 1e-30)

err_M = 0.0
for _frac in (0.15, 0.25, 0.35):
    _i = int(np.argmin(np.abs(x_vals - _frac * L_beam)))
    _tot = (sum(Mi_vals[k][_i] for k in range(n))
            + sum(Ni_vals[k][_i] * (zG_chk - z0[k]) for k in range(n)))
    if abs(M_vals[_i]) > 1.0:
        err_M = max(err_M, abs(_tot - M_vals[_i]) / abs(M_vals[_i]))

print(f"Step 5: Consistency Check - sum(Ni) = {err_N:.3e} ; sectional equilibrium = {err_M:.3e}"
      + (" --- OK" if (err_N < 1e-6 and err_M < 1e-3) else " --- KO"))

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


