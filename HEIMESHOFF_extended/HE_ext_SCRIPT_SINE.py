# ===============================================================
# FUNCTIONS & IMPORTS
# ===============================================================

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from HE_ext_FUNCTIONS import (read_data, HE_ext_Parameters, HE_ext_Rigidity,
                              HE_ext_SigmaSection, HE_ext_SigmaMaxi,
                              HE_ext_TauSection, HE_ext_TauMaxi, HE_ext_Ts,
                              HE_ext_ConnectorForces)


# ===============================================================
# INPUT DATA
# ===============================================================
## NAME CONFIG
name = 'HE-ext_SINE'

## LAMELLAE PARAMETERS
L_beam = 1400.0         # Beam length [mm]

## LOAD PARAMETERS
q0 = 1                 # [N/mm]


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
F_position = L_beam/2
l = L_beam                  # Length of load application [mm]
d = F_position - l/2        # Length btw support and load [mm]
e = F_position + l/2



## PRINT DATA
print(f'============================================ {name} ==============================================')
print(f'Force : qm = {q0} [N]')
print(f'Length of the beam : L = {L_beam} [mm]')
print(f'Position of the force : F_pos = {F_position} [mm]')
print('\n')

print(f'Length of load application : l = {l} [mm]')
print(f'Linear load : qm = {q0} [N/mm]')
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


# ===============================================================
# STEP 0 — COMPUTE BEAM PARAMETERS AND RIGIDITIES
# ===============================================================
out = HE_ext_Parameters(LAM, CON, L_beam)

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
a = out['a']
zGC = out['zGC']

# 'a' is reassigned from the return value on purpose: HE_ext_Rigidity offsets
# any lamella sitting exactly on the neutral axis by a small eps (a fraction of
# that lamella's own height) to make the system regular, and every downstream
# quantity - Ni, z, EIeff, stresses, connector forces - must use that same
# vector. Using the raw geometric a here instead would multiply a diverging
# Gamma_i by an exact zero and silently drop that lamella's axial force,
# breaking sum_i Ni = 0 (the consistency check in Step 4 catches this).
Gamma, z, EIeff, a = HE_ext_Rigidity(n, c, s, a, EA, EI, L_beam, h=h)

print(f' -> Position from zGC : ai = {a}')
print(f' -> Reduction coefficient : γ = {Gamma}')
print(f' -> Effective flexural rigidity : EIeff = {EIeff}')

print("Step 0: Parameters and rigidities --- OK")


# ===============================================================
# STEP 1 — APPLIED LOAD & INTERNAL FORCES - COMPUTE & EXPORT
# ===============================================================
out_folder = "RES_all"
os.makedirs(out_folder, exist_ok=True)

def q_expr(x):
    """Sinusoidal load"""
    return np.full_like(np.atleast_1d(x), q0 * np.sin(omega * x), dtype=float)
q_vals = q_expr(x_vals)

def w_expr(x):
    return np.full_like(np.atleast_1d(x), q0 * 1/EIeff * 1/(omega ** 4) * np.sin(omega * x), dtype=float)
w_vals = w_expr(x_vals)

def M_expr(x):
    return np.full_like(np.atleast_1d(x), q0 * 1/(omega ** 2) * np.sin(omega * x), dtype=float)
M_vals = M_expr(x_vals)

def T_expr(x):
    return np.full_like(np.atleast_1d(x), q0  * 1/(omega) * np.cos(omega * x), dtype=float)
T_vals = T_expr(x_vals)

Mi_vals = [M_vals * EI[i] /EIeff for i in range(n)]
Ni_vals = [-M_vals * Gamma[i] * EA[i] * a[i] / EIeff for i in range(n)]

data ={
    'x [mm]': x_vals,
    'q [N/mm]': q_vals,
    'w_HE [mm]': w_vals,
    'M [Nmm]': M_vals,
    'T [N]': T_vals
}
for i in range(n):
    data[f'M{i+1} [Nmm]'] = Mi_vals[i]

for i in range(n):
    data[f'N{i + 1} [N]'] = Ni_vals[i]

df_all = pd.DataFrame(data)
filename = os.path.join(out_folder, f"all_{name}.csv")
df_all.to_csv(filename, sep="\t", index=False)

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

i_wmax = int(np.argmax(np.abs(w_vals)))
print(f' -> Maximal deflection : wmax = {w_vals[i_wmax]} [mm] at x = {x_vals[i_wmax]} [mm]')
print("Step 1: Internal forces --- OK")

# ===============================================================
# STEP 2 — STRESS COMPUTATION FOR BEAM
# ===============================================================
out_folder = "RES_stresses"
os.makedirs(out_folder, exist_ok=True)


# Normal
M = [float(np.interp(x0_sigma, x_vals, Mi_vals[i])) for i in range(n)]
N = [float(np.interp(x0_sigma, x_vals, Ni_vals[i])) for i in range(n)]

df_sigmaSection = HE_ext_SigmaSection(M, N, h, b, z0)

filename = os.path.join(out_folder, f"stresses_{name}_SIGMA_zglobal.csv")
df_sigmaSection.to_csv(filename, sep="\t", index=False)

df_sigmaMAX_L = pd.DataFrame({'x [mm]': x_vals})
for i in range(n):
    sigmaTOT_max_list = []
    for val in x_vals:
        M_i = float(np.interp(val, x_vals, Mi_vals[i]))
        N_i = float(np.interp(val, x_vals, Ni_vals[i]))
        sigmaTOT_max_list.append(HE_ext_SigmaMaxi(i, M_i, N_i, h, b))
    df_sigmaMAX_L[f'σTOT_{i + 1} [MPa]'] = sigmaTOT_max_list

filename = os.path.join(out_folder, f"stresses_{name}_SIGMA_MAX_L.csv")
df_sigmaMAX_L.to_csv(filename, sep="\t", index=False)


# Shear
T = float(np.interp(x0_tau, x_vals, T_vals))
df_tauSection = HE_ext_TauSection (T, h, b, E, z, a, z0, c, EIeff, idx_inf, idx_cent, idx_sup)

filename = os.path.join(out_folder, f"stresses_{name}_TAU_zglobal.csv")
df_tauSection.to_csv(filename, sep="\t", index=False)

tau_max_all = []
for val in x_vals:
    T_i = float(np.interp(val, x_vals, T_vals))
    tau_max_i = HE_ext_TauMaxi(T_i, h, b, E, z, a, z0, c, EIeff, idx_inf, idx_cent, idx_sup)
    tau_max_all.append(tau_max_i)

df_tauMAX_L = pd.DataFrame(tau_max_all, columns=[f"τTOT_{i+1} [MPa]" for i in range(n)])
df_tauMAX_L.insert(0, "x [mm]", x_vals)

filename = os.path.join(out_folder, f"stresses_{name}_TAU_MAX_L.csv")
df_tauMAX_L.to_csv(filename, sep="\t", index=False)

print("Step 2: Beam stresses --- OK")

# ===============================================================
# STEP 3 — SHEAR FORCES IN CONNECTORS
# ===============================================================
out_folder = "RES_interface"
os.makedirs(out_folder, exist_ok=True)

df_Ts = pd.DataFrame({'x [mm]': x_vals})

Ts_all = []
for val in x_vals:
    T_i = float(np.interp(val, x_vals, T_vals))
    Ts_i = HE_ext_Ts(T_i, h, b, E, z, a, z0, c, EIeff, idx_inf, idx_cent, idx_sup)
    Ts_all.append(Ts_i)

df_Ts = pd.DataFrame(Ts_all, columns=[f"Ts_{i+1},{i+2} [N/mm]" for i in range(n-1)])
df_Ts.insert(0, "x [mm]", x_vals)
filename = os.path.join(out_folder, f"Ts_{name}.csv")
df_Ts.to_csv(filename, sep="\t", index=False)


df_Fcon, Fconn_max = HE_ext_ConnectorForces(x_vals, df_Ts, L_beam, s)
filename = os.path.join(out_folder, f"Fconn_{name}.csv")
df_Fcon.to_csv(filename, sep='\t', index=False)

for _j, _col in enumerate([c for c in df_Ts.columns if c != 'x [mm]']):
    print(f" -> Maximum connector load - {_col.replace('Ts_', 'Fconn').split(' [')[0]} :"
          f" {Fconn_max[_j]} [N]")

print("Step 3: Shear Forces in connectors --- OK")
# ===============================================================
# STEP 4 — CONSISTENCY CHECK OF RESULTS
# Sectional equilibrium of the layered cross-section:
#     M(x) = sum_i M_i(x) + sum_i N_i(x) * (zG - z_i)
# zG is the elastic centroid; since sum_i N_i = 0 the result does not
# depend on the reference level, but zG keeps both terms of comparable
# magnitude. Signs follow the sigma = -M*z/I convention, identical to the
# GIRHAMMAR, GIRHAMMAR_generalised and HEIMESHOFF methods.
# ===============================================================
zG_chk = np.sum(z0 * EA) / np.sum(EA)
err_N = np.max(np.abs(sum(Ni_vals[i] for i in range(n)))) / max(np.max(np.abs(Ni_vals[0])), 1e-30)

err_M = 0.0
for _frac in (0.15, 0.25, 0.35):
    _i = int(np.argmin(np.abs(x_vals - _frac * L_beam)))
    _tot = (sum(Mi_vals[k][_i] for k in range(n))
            + sum(Ni_vals[k][_i] * (zG_chk - z0[k]) for k in range(n)))
    if abs(M_vals[_i]) > 1.0:
        err_M = max(err_M, abs(_tot - M_vals[_i]) / abs(M_vals[_i]))

print(f"Step 4: Consistency Check - sum(Ni) = {err_N:.3e} ; sectional equilibrium = {err_M:.3e}"
      + (" --- OK" if (err_N < 1e-6 and err_M < 1e-6) else " --- KO"))

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


